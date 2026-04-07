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

# =====================================================================
# 基础配置与日志
# =====================================================================
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

HISTORY_FILE = "processed_history.json"
ONLY_TOP_CONF = os.getenv("ONLY_TOP_CONF", "True").lower() in ("true", "1", "yes")

# 检查必要环境变量
def check_env():
    missing = []
    if not os.getenv("DEEPSEEK_API_KEY"): missing.append("DEEPSEEK_API_KEY")
    if not os.getenv("FEISHU_WEBHOOK"): missing.append("FEISHU_WEBHOOK")
    if missing:
        logging.warning(f"⚠️ 缺少环境变量: {', '.join(missing)}。脚本将以受限模式运行。")
    return missing

# =====================================================================
# 历史记录管理
# =====================================================================
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
# PDF 解析模块
# =====================================================================
def extract_pdf_core_content(pdf_url):
    try:
        pdf_dl_url = pdf_url.replace('abs', 'pdf') + ".pdf"
        logging.info(f"📄 正在解析 PDF: {pdf_dl_url}")
        time.sleep(2) # 礼貌延迟
        
        r = requests.get(pdf_dl_url, timeout=30)
        r.raise_for_status()
        
        with fitz.open(stream=r.content, filetype="pdf") as doc:
            text = ""
            if len(doc) > 0:
                text += f"【Intro】:\n{doc[0].get_text()[:1200]}\n"
            if len(doc) > 1:
                text += f"【Conclusion】:\n{doc[-1].get_text()[-1200:]}\n"
            return text
    except Exception as e:
        logging.warning(f"⚠️ PDF解析失败: {e}")
        return "无法提取PDF正文，请基于摘要评估。"

# =====================================================================
# DeepSeek 评估引擎
# =====================================================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_deepseek_api(payload, headers):
    resp = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

def evaluate_and_rank_with_deepseek(candidates, topic):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or not candidates: 
        return candidates[:3] # 无Key则返回前3

    prompt_text = f"你是一个资深的【{topic}】领域AI研究员。请评估以下最新ArXiv论文。\n"
    prompt_text += "挑选出最优秀的最多 5 篇，并进行深度解析。如果是自动驾驶、规控、RL相关则优先。\n\n"
    
    for paper in candidates:
        prompt_text += f"ID: {paper['paper_key']} | Title: {paper['title']}\n"
        prompt_text += f"Abstract: {paper['abstract']}\n"
        prompt_text += f"PDF Sample: {paper.get('pdf_text', '')[:500]}\n---\n"
    
    prompt_text += """
严格输出 JSON 格式（不要Markdown代码块包裹），结构如下：
{
  "top_papers": [
    {
      "id": "论文ID",
      "score": 95,
      "tags": ["核心技术标签"],
      "github_link": "URL或null",
      "review": {
        "type": "方向定性",
        "pain_point": "解决的痛点",
        "innovation": "核心创新点",
        "comparison": "性能对比",
        "scenario": "应用场景",
        "advice": "行动建议",
        "layman_summary": "通俗大白话"
      }
    }
  ]
}
"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个只输出JSON的严谨学术助手。"},
            {"role": "user", "content": prompt_text}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3
    }

    try:
        logging.info(f"🧠 AI 正在评审 {len(candidates)} 篇文献...")
        result_json = call_deepseek_api(payload, headers)
        content = result_json['choices'][0]['message']['content']
        ai_evaluations = json.loads(content).get("top_papers", [])
        
        evaluated_papers = []
        for ai_eval in ai_evaluations:
            for paper in candidates:
                if paper['paper_key'] == ai_eval.get('id'):
                    paper.update({
                        'ai_score': ai_eval.get('score', 0),
                        'tags': ai_eval.get('tags', []),
                        'github_link': ai_eval.get('github_link'),
                        'review': ai_eval.get('review', {})
                    })
                    evaluated_papers.append(paper)
                    break
        return evaluated_papers
    except Exception as e:
        logging.error(f"❌ AI 评估失败: {e}")
        return []

# =====================================================================
# 飞书推送模块
# =====================================================================
def send_to_feishu(webhook, paper):
    if not webhook: return
    
    title = paper['title']
    url = paper['url']
    review = paper.get('review', {})
    tags_str = " ".join([f"**[{t}]**" for t in paper.get('tags', [])])

    md_content = (
        f"👤 **作者**: {paper['authors']}\n"
        f"🏷️ **标签**: {tags_str}\n"
        f"🔥 **AI 评分**: {paper.get('ai_score', 'N/A')}\n"
    )
    if paper.get('github_link') and str(paper['github_link']).lower() != "null":
        md_content += f"💻 **代码**: [GitHub]({paper['github_link']})\n"
    
    md_content += (
        f"\n---\n"
        f"**🎯 定性**: {review.get('type','-')}\n"
        f"**💢 痛点**: {review.get('pain_point','-')}\n"
        f"**✨ 创新**: {review.get('innovation','-')}\n"
        f"**🚗 场景**: {review.get('scenario','-')}\n"
        f"**📌 建议**: {review.get('advice','-')}\n"
        f"\n---\n"
        f"🗣️ **通俗总结**: *{review.get('layman_summary','-')}*\n"
        f"\n🔗 [查看原文]({url})"
    )

    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"🌟 {paper['topic']} Top Pick: {title[:60]}..."},
                "template": "orange" if paper.get('ai_score', 0) > 90 else "blue"
            },
            "elements": [{"tag": "markdown", "content": md_content}]
        }
    }
    
    try:
        r = requests.post(webhook, json=payload, timeout=10)
        r.raise_for_status()
        logging.info(f"✅ 飞书推送成功: {title[:30]}")
    except Exception as e:
        logging.error(f"❌ 飞书推送失败: {e}")

# =====================================================================
# 业务逻辑流
# =====================================================================
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 处理关键词逻辑
    keywords = {}
    for k, v in config.get('keywords', {}).items():
        filters = v.get('filters', [])
        keywords[k] = " OR ".join([f'"{f}"' if " " in f else f for f in filters])
    config['kv'] = keywords
    return config

def get_papers(topic, query, max_results):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    papers = []
    for result in client.results(search):
        p_id = result.get_short_id().split('v')[0]
        if p_id in PROCESSED_HISTORY: continue
        if ONLY_TOP_CONF and not is_top_conf(result.comment): continue

        paper_url = f"http://arxiv.org/abs/{p_id}"
        pdf_text = extract_pdf_core_content(paper_url)
        
        papers.append({
            'topic': topic,
            'paper_key': p_id,
            'title': result.title,
            'url': paper_url,
            'abstract': result.summary,
            'pdf_text': pdf_text,
            'authors': ", ".join(str(a) for a in result.authors),
            'first_author': str(result.authors[0]) if result.authors else "Unknown",
            'update_time': str(result.updated.date()),
            'comments': result.comment
        })
    return papers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml')
    args = parser.parse_args()
    
    check_env()
    config = load_config(args.config_path)
    webhook = os.getenv("FEISHU_WEBHOOK")
    
    all_candidates = []
    for topic, query in config['kv'].items():
        logging.info(f"🔍 正在检索领域: {topic}")
        papers = get_papers(topic, query, config.get('max_results', 10))
        if papers:
            # 每个领域单独评估，保证领域内公平
            evaluated = evaluate_and_rank_with_deepseek(papers, topic)
            all_candidates.extend(evaluated)
    
    # 全局按分数排序
    all_candidates.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
    top_picks = all_candidates[:5] # 最终精选 5 篇推送

    if not top_picks:
        logging.info("📭 今日无新论文或未通过 AI 筛选。")
        return

    for paper in top_picks:
        send_to_feishu(webhook, paper)
        PROCESSED_HISTORY.add(paper['paper_key'])
    
    save_history(PROCESSED_HISTORY)
    logging.info("🏁 任务完成，历史记录已更新。")

if __name__ == "__main__":
    main()