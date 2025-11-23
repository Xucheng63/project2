"""
review_bridge.py - 完全修正版
将 AI-Scientist 生成的 PDF 论文转换为 Markdown 并调用 OpenReviewer 生成评审
"""

import argparse
import sys
from pathlib import Path

# 适配目录结构：ai-scientist/AI-Scientist/tools/ 和 ai-scientist/openreviewer/
project_root = Path(__file__).parent.parent.parent
openreviewer_path = project_root / 'openreviewer'

print(f"🔍 路径调试:")
print(f"   当前脚本: {Path(__file__)}")
print(f"   项目根目录: {project_root}")
print(f"   OpenReviewer 路径: {openreviewer_path}")
print(f"   OpenReviewer 存在: {openreviewer_path.exists()}\n")

sys.path.insert(0, str(openreviewer_path))

try:
    from app import generate
    print("✅ 成功导入 OpenReviewer\n")
except ImportError as e:
    print(f"❌ 无法导入 openreviewer.app: {e}")
    print(f"请确保 openreviewer 目录在: {openreviewer_path}")
    sys.exit(1)


def pdf_to_markdown(pdf_path: str) -> str:
    """使用 PyMuPDF 提取 PDF 文本"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("❌ PyMuPDF 未安装")
        print("请运行: pip install PyMuPDF")
        sys.exit(1)
    
    print(f"📄 提取 PDF 文本: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    page_count = len(doc)  # ← 在关闭前保存页数
    markdown_text = ""
    
    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text()
        markdown_text += text + "\n\n"
    
    doc.close()
    
    # 在关闭后使用保存的 page_count
    print(f"✅ 提取完成: {page_count} 页，共 {len(markdown_text)} 字符")
    return markdown_text.strip()


def run_openreviewer(text_md: str, show_progress: bool = False) -> str:
    """调用 OpenReviewer 生成评审意见"""
    
    REVIEW_TEMPLATE = """## Summary
Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.

## Novelty
Please assign the paper a numerical rating on the following scale to indicate the novelty and originality of the work. Consider whether the paper presents new ideas, methods, or perspectives that have not been explored before. Choose from the following:
4: excellent - Highly original work with groundbreaking ideas or completely novel approaches
3: good - Significant new contributions with clear advances over existing work
2: fair - Some new elements but largely incremental improvements or combinations of existing ideas
1: poor - Little to no novelty, mostly reproducing existing work or trivial variations

## Novelty Explanation
IMPORTANT: Focus ONLY on novelty aspects. DO NOT discuss soundness, presentation, or general contribution here.
Please provide specific justification for your novelty score by addressing:
- What specific new concepts, methods, or approaches does this paper introduce?
- How do these differ from existing work in the field? Cite specific prior work for comparison.
- Are the differences substantial or incremental?
- Is this addressing a problem in a genuinely new way, or applying known methods to a new domain?
DO NOT repeat content from other sections. DO NOT discuss writing quality, experimental rigor, or implementation details here.

## Soundness
Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence. Choose from the following:
4: excellent
3: good
2: fair
1: poor

## Soundness Explanation
IMPORTANT: Focus ONLY on technical correctness and methodological rigor. DO NOT discuss novelty or writing quality here.
Please provide specific reasons for your soundness score by addressing:
- Are the technical claims mathematically/logically correct?
- Is the experimental methodology rigorous and appropriate?
- Are the experiments sufficient to support the claims?
- Are there any methodological flaws or missing controls?
- Is the statistical analysis (if any) appropriate and correctly executed?
DO NOT repeat content from other sections. DO NOT discuss the novelty of the approach or presentation quality here.

## Presentation
Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work. Choose from the following:
4: excellent
3: good
2: fair
1: poor

## Presentation Explanation
IMPORTANT: Focus ONLY on writing quality, clarity, and organization. DO NOT discuss technical merit or novelty here.
Please explain your presentation score by addressing:
- Is the paper well-organized and easy to follow?
- Are the main ideas clearly explained?
- Are figures, tables, and visualizations effective and well-designed?
- Is the related work section comprehensive and fair?
- Are mathematical notations consistent and clear?
- Is the language precise and grammatically correct?
DO NOT repeat content from other sections. DO NOT discuss the novelty of ideas or soundness of methods here.

## Contribution
Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader ICLR community? Choose from the following:
4: excellent
3: good
2: fair
1: poor

## Contribution Explanation
IMPORTANT: Focus on the OVERALL IMPACT and SIGNIFICANCE to the field. This is different from novelty.
Please justify your contribution score by explaining:
- Why is this work important for the field?
- What practical or theoretical impact could this have?
- Who would benefit from this work and how?
- Does this open new research directions or close important gaps?
- How significant are the improvements over baselines (if applicable)?
Consider both immediate utility and long-term impact. DO NOT simply repeat the novelty assessment here.

## Strengths
List the main strengths of the paper. Be specific and provide evidence. Each strength should be a separate bullet point. Focus on what the paper does well across all dimensions (novelty, soundness, presentation, contribution). Avoid generic statements.

## Weaknesses
List the main weaknesses of the paper. Be specific, constructive, and actionable. Each weakness should be a separate bullet point with suggestions for improvement where possible. Focus on significant issues that affect the paper's validity or impact.

## Questions
List specific questions for the authors that could clarify ambiguities or address concerns. Number each question. These should be questions where the answer could potentially change your assessment of the paper.

## Flag For Ethics Review
If there are ethical issues with this paper, please flag the paper for an ethics review and select area of expertise that would be most useful for the ethics reviewer to have. Please select all that apply. Choose from the following:
No ethics review needed.
Yes, Discrimination / bias / fairness concerns
Yes, Privacy, security and safety
Yes, Legal compliance (e.g., GDPR, copyright, terms of use)
Yes, Potentially harmful insights, methodologies and applications
Yes, Responsible research practice (e.g., human subjects, data release)
Yes, Research integrity issues (e.g., plagiarism, dual submission)
Yes, Unprofessional behaviors (e.g., unprofessional exchange between authors and reviewers)
Yes, Other reasons (please specify below)

## Details Of Ethics Concerns
Please provide details of your concerns. If no ethics review is needed, write "N/A".

## Rating
Please provide an "overall score" for this submission. Choose from the following:
1: strong reject
3: reject, not good enough
5: marginally below the acceptance threshold
6: marginally above the acceptance threshold
8: accept, good paper
10: strong accept, should be highlighted at the conference

## Overall Justification
Provide a comprehensive justification for your overall rating that:
- Synthesizes the assessments from all dimensions (novelty, soundness, presentation, contribution)
- Explains how you weighted different aspects in arriving at your final score
- Clearly states whether the strengths outweigh the weaknesses or vice versa
- Indicates what would need to change for a different rating
This should be a holistic assessment, not a repetition of individual sections.
"""
    
    print("🔍 OpenReviewer 正在生成评审意见...")
    if show_progress:
        print("   (实时显示进度)\n")
    
    try:
        output = generate(text_md, review_template=REVIEW_TEMPLATE)
        
        # 带进度显示
        if show_progress:
            collected = []
            last_length = 0
            
            if hasattr(output, "__iter__") and not isinstance(output, (str, dict)):
                for item in output:
                    if isinstance(item, str):
                        collected.append(item)
                        current_text = "\n".join(collected)
                        new_text = current_text[last_length:]
                        if new_text:
                            print(new_text, end='', flush=True)
                            last_length = len(current_text)
                
                print("\n")
                return "\n".join(collected)
        
        # 不显示进度
        if isinstance(output, str):
            return output
        
        if hasattr(output, "__iter__") and not isinstance(output, (str, dict)):
            collected = []
            for item in output:
                if isinstance(item, str):
                    collected.append(item)
            return "\n".join(collected)
        
        if isinstance(output, dict):
            md = ""
            for k, v in output.items():
                md += f"## {k}\n{v}\n\n"
            return md
        
        return str(output)
    
    except Exception as e:
        print(f"❌ OpenReviewer 生成失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="将 AI-Scientist 生成的 PDF 论文自动评审"
    )
    parser.add_argument("--pdf", required=True, help="输入的 PDF 文件路径")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--model", default="gpt-4o", help="模型名称（保留参数）")
    parser.add_argument("--show-progress", action="store_true", help="显示生成进度")
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not pdf_path.exists():
        print(f"❌ PDF 文件不存在: {pdf_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🚀 开始处理论文评审")
    print("="*70)
    print(f"📄 PDF: {pdf_path}")
    print(f"📁 输出: {out_dir}")
    print("="*70 + "\n")
    
    # Step 1: PDF → Markdown
    markdown_text = pdf_to_markdown(str(pdf_path))
    md_path = out_dir / "paper.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    print(f"✅ Markdown: {md_path}\n")
    
    # Step 2: 生成评审
    review_md = run_openreviewer(markdown_text, show_progress=args.show_progress)
    review_path = out_dir / "review.md"
    review_path.write_text(review_md, encoding="utf-8")
    print(f"✅ 评审: {review_path}\n")
    
    print("="*70)
    print("🎉 完成！")
    print("="*70)


if __name__ == "__main__":
    main()