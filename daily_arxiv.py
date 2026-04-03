import os
import re
import time
import json
import arxiv
import yaml
import logging
import argparse
import datetime
import requests
import fitz  # PyMuPDF
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

github_url = "https://api.github.com/search/repositories"
arxiv_url = "http://arxiv.org/"

HISTORY_FILE = "processed_history.json"
ONLY_TOP_CONF = os.getenv("ONLY_TOP_CONF", "True").lower() in ("true", "1", "yes")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            logging.error(f"加载历史记录失败: {e}")
    return set()

def save_history(history_set):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(history_set), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"保存历史记录失败: {e}")

PROCESSED_HISTORY = load_history()

def is_top_conf(comment_text):
    if not comment_text: return False
    conf_pattern = re.compile(r'(ICRA|IROS|NeurIPS|ICML|AAAI|TRO|IJRR|RSS|CoRL|CVPR|ICCV|ECCV|RAL)', re.IGNORECASE)
    return bool(conf_pattern.search(comment_text))

# =====================================================================
# [新增] PDF 干货提取模块
# =====================================================================
def extract_pdf_core_content(pdf_url):
    """下载并提取论文第一页(Intro)和最后一页(Conclusion)"""
    try:
        pdf_dl_url = pdf_url.replace('abs', 'pdf') + ".pdf"
        logging.info(f"📄 正在解析 PDF (礼貌延迟3秒防封): {pdf_dl_url}")
        time.sleep(3) # 防止被 arXiv 拉黑 IP
        
        r = requests.get(pdf_dl_url, timeout=20)
        r.raise_for_status()
        
        doc = fitz.open(stream=r.content, filetype="pdf")
        text = ""
        if len(doc) > 0:
            text += f"【Introduction】:\n{doc[0].get_text()[:1500]}\n" # 截取前1500字符防超载
        if len(doc) > 1:
            text += f"【Conclusion】:\n{doc[-1].get_text()[-1500:]}\n"
        doc.close()
        return text
    except Exception as e:
        logging.warning(f"⚠️ PDF解析失败, 回退至摘要模式: {e}")
        return "PDF解析失败，请仅依赖摘要进行评估。"

# =====================================================================
# [重构] DeepSeek 评估模块 (Tenacity 重试 + 新 Prompt)
# =====================================================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_deepseek_api(payload, headers):
    """封装 API 请求以支持失败自动重试"""
    resp = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=40)
    resp.raise_for_status()
    return resp.json()

def evaluate_and_rank_with_deepseek(candidates, topic):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key: return candidates[:5]
    if not candidates: return []

    prompt_text = f"你是一个资深的【{topic}】领域AI研究员。请评估以下最新ArXiv论文候选列表。\n"
    prompt_text += "请根据论文的【创新性、质量、对该领域的相关性和影响力】进行打分(0-100分)，并挑选出最优秀的最多 5 篇论文。\n\n"
    prompt_text += "候选论文列表：\n"
    
    for paper in candidates:
        prompt_text += f"ID: {paper['paper_key']} | Title: {paper['title']}\n"
        prompt_text += f"Abstract: {paper['abstract']}\n"
        prompt_text += f"PDF Text: {paper.get('pdf_text', '')}\n"
        prompt_text += f"Comments: {paper.get('comments', '')}\n---\n"
    
    prompt_text += """
请严格以 JSON 格式输出结果，格式如下：
{
  "top_papers": [
    {
      "id": "此处填写论文ID",
      "score": 95,
      "ai_summary": "一句话中文锐评核心贡献",
      "tags": ["Tag1", "Tag2"],  // 生成2-3个英文技术微标签，如 [Sim2Real], [RL]
      "github_link": "提取文本中的GitHub链接，如果没有必须返回 null"
    }
  ]
}
"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个严格的学术机器人。只输出合法的 JSON 字符串。"},
            {"role": "user", "content": prompt_text}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }

    try:
        logging.info(f"🧠 调用 DeepSeek 评估 {len(candidates)} 篇 [{topic}] 领域候选论文...")
        result_json = call_deepseek_api(payload, headers)
        content = result_json['choices'][0]['message']['content']
        
        ai_evaluations = json.loads(content).get("top_papers", [])
        final_top_papers = []
        for ai_eval in sorted(ai_evaluations, key=lambda x: x.get("score", 0), reverse=True):
            for paper in candidates:
                if paper['paper_key'] == ai_eval.get('id'):
                    paper['ai_score'] = ai_eval.get('score', 0)
                    paper['ai_summary'] = ai_eval.get('ai_summary', "AI暂无评价")
                    paper['tags'] = ai_eval.get('tags', [])
                    paper['github_link'] = ai_eval.get('github_link')
                    final_top_papers.append(paper)
                    break
        return final_top_papers
    except Exception as e:
        logging.error(f"❌ DeepSeek 调用或解析在多次重试后仍失败: {e}")
        return candidates[:5]

# =====================================================================
# [重构] 飞书富文本推送模块 (支持 Tag 和 GitHub 高亮)
# =====================================================================
def send_to_feishu(webhook, paper):
    if not webhook: return
    
    topic = paper.get('topic', '未知领域')
    title = paper['title']
    url = paper['url']
    ai_score = paper.get('ai_score', 'N/A')
    ai_summary = paper.get('ai_summary', '无')
    tags = paper.get('tags', [])
    github_link = paper.get('github_link')
    
    # 构造多彩标签
    tags_str = " ".join([f"**[{t}]**" for t in tags]) if tags else "无"
    
    # 构建飞书 Markdown
    md_content = f"👤 **作者**: {paper['authors']}\n"
    md_content += f"🏷️ **分类**: {tags_str}\n"
    if github_link and github_link.lower() != "null":
        md_content += f"💻 **开源代码**: [{github_link}]({github_link})\n"
    md_content += f"🔥 **AI 评分**: {ai_score} / 100\n"
    md_content += f"💡 **AI 锐评**: {ai_summary}\n"
    md_content += f"🔗 **原件链接**: [{url}]({url})"

    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"🏆 [全场 Top 5 | {topic}]\n{title[:50]}..."},
                "template": "blue" if github_link and github_link.lower() != "null" else "wathet" 
            },
            "elements": [
                {"tag": "markdown", "content": md_content}
            ]
        }
    }
    
    try:
        r = requests.post(webhook, json=payload, headers={'Content-Type': 'application/json'}, timeout=5)
        if r.status_code == 200: logging.info(f"✅ 飞书推送成功: {title[:30]}")
    except Exception as e:
        logging.error(f"❌ 飞书推送发生异常: {e}")

# =====================================================================
# 核心调度流保持不变，仅在拉取时注入 PDF 解析
# =====================================================================
def load_config(config_file:str) -> dict:
    def pretty_filters(**config) -> dict:
        keywords = dict()
        def parse_filters(filters:list):
            return " OR ".join([f'"{f}"' if len(f.split()) > 1 else f for f in filters])
        for k,v in config['keywords'].items(): keywords[k] = parse_filters(v['filters'])
        return keywords
    with open(config_file,'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        config['kv'] = pretty_filters(**config)
    return config

def get_authors(authors, first_author=False):
    return authors[0] if first_author else ", ".join(str(author) for author in authors)

def get_and_evaluate_papers(topic, query, max_results):
    client = arxiv.Client()
    search_engine = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    candidates = []
    for result in client.results(search_engine):
        paper_id = result.get_short_id()
        paper_key = paper_id.split('v')[0]
        
        if paper_key in PROCESSED_HISTORY: continue
        if ONLY_TOP_CONF and not is_top_conf(result.comment): continue

        paper_url = arxiv_url + 'abs/' + paper_key
        # 下载并提取 PDF 关键文本
        pdf_text = extract_pdf_core_content(paper_url)

        candidates.append({
            'topic': topic,
            'paper_key': paper_key,
            'title': result.title,
            'url': paper_url,
            'abstract': result.summary,
            'pdf_text': pdf_text,
            'authors': get_authors(result.authors),
            'first_author': get_authors(result.authors, first_author=True),
            'update_time': str(result.updated.date()),
            'comments': result.comment
        })
        logging.info(f"📥 提取有效文献: [{paper_key}] {result.title[:30]}...")

    for paper in candidates: PROCESSED_HISTORY.add(paper['paper_key'])

    if candidates: return evaluate_and_rank_with_deepseek(candidates, topic)
    return []

# （此处省略 update_json_file 和 json_to_md 等无须更改的写库函数，保持上一版的写法即可）
def update_json_file(filename, data_dict):
    m = json.loads(open(filename).read()) if os.path.exists(filename) and os.path.getsize(filename) > 0 else {}
    for data in data_dict:
        for k, v in data.items():
            m.setdefault(k, {}).update(v)
    with open(filename, "w") as f: json.dump(m, f)

def json_to_md(filename, md_filename):
    data = json.loads(open(filename).read()) if os.path.exists(filename) else {}
    DateNow = str(datetime.date.today()).replace('-','.')
    with open(md_filename, "w+") as f:
        f.write(f"## Updated on {DateNow}\n> Usage instructions: [here](./docs/README.md#usage)\n\n")
        f.write("<details>\n  <summary>Table of Contents</summary>\n  <ol>\n")
        for k in data:
            if data[k]: f.write(f"    <li><a href=#{k.replace(' ','-').lower()}>{k}</a></li>\n")
        f.write("  </ol>\n</details>\n\n")
        for k, v in data.items():
            if not v: continue
            f.write(f"## {k}\n\n| Publish Date | Title | Authors | PDF | Code |\n|:---------|:-----------------------|:---------|:------|:------|\n")
            keys = list(v.keys()); keys.sort(reverse=True)
            for key in keys: f.write(v[key])
            f.write("\n")

def demo(**config):
    keywords = config['kv']
    max_results = config['max_results']
    global_top_papers = []

    logging.info(f"=== 开始拉取 arXiv 文献并提交 AI 评估 ===")
    for topic, keyword in keywords.items():
        logging.info(f"正在处理领域: {topic}")
        topic_evaluated = get_and_evaluate_papers(topic, keyword, max_results)
        global_top_papers.extend(topic_evaluated)
    
    global_top_papers.sort(key=lambda x: float(x.get('ai_score', 0)), reverse=True)
    top_5_to_push = global_top_papers[:5]

    save_history(PROCESSED_HISTORY)
    webhook = os.getenv("FEISHU_WEBHOOK")
    
    data_collector = []
    for topic in keywords.keys():
        content = dict()
        for paper in top_5_to_push:
            if paper['topic'] == topic:
                if webhook: send_to_feishu(webhook, paper)
                line = "- {}, **{}**, {} et.al., Paper: [{}]({})".format(
                   paper['update_time'], paper['title'], paper['first_author'], paper['url'], paper['url'])
                if paper['comments']: line += f", {paper['comments'].replace(chr(10), ' ')}"
                content[paper['paper_key']] = line + "\n"
        data_collector.append({topic: content})

    if config.get('publish_readme'):
        json_file = config['json_readme_path']
        md_file = config['md_readme_path']
        update_json_file(json_file, data_collector)
        json_to_md(json_file, md_file)
        logging.info(f"📚 本地精华仓库更新完毕。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str, default='config.yaml')
    args = parser.parse_args()
    demo(**load_config(args.config_path))