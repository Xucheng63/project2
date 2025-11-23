import argparse
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
from typing import Optional, Tuple
from pathlib import Path

from ai_scientist.generate_ideas import search_for_papers
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS


# GENERATE LATEX
def generate_latex(coder, folder_name, pdf_file, timeout=30, num_error_corrections=5):
    folder = osp.abspath(folder_name)
    cwd = osp.join(folder, "latex")  # Fixed potential issue with path
    writeup_file = osp.join(cwd, "template.tex")

    # Check all references are valid and in the references.bib file
    with open(writeup_file, "r") as f:
        tex_text = f.read()
    cites = re.findall(r"\\cite[a-z]*{([^}]*)}", tex_text)
    references_bib = re.search(
        r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}",
        tex_text,
        re.DOTALL,
    )
    if references_bib is None:
        print("No references.bib found in template.tex")
        return
    bib_text = references_bib.group(1)
    cites = [cite.strip() for item in cites for cite in item.split(",")]
    for cite in cites:
        if cite not in bib_text:
            print(f"Reference {cite} not found in references.")
            prompt = f"""Reference {cite} not found in references.bib. Is this included under a different name?
If so, please modify the citation in template.tex to match the name in references.bib at the top. Otherwise, remove the cite."""
            coder.run(prompt)

    # Check all included figures are actually in the directory.
    with open(writeup_file, "r") as f:
        tex_text = f.read()
    referenced_figs = re.findall(r"\\includegraphics.*?{(.*?)}", tex_text)
    all_figs = [f for f in os.listdir(folder) if f.endswith(".png")]
    for figure in referenced_figs:
        if figure not in all_figs:
            print(f"Figure {figure} not found in directory.")
            prompt = f"""The image {figure} not found in the directory. The images in the directory are: {all_figs}.
Please ensure that the figure is in the directory and that the filename is correct. Check the notes to see what each figure contains."""
            coder.run(prompt)

    # Remove duplicate figures.
    with open(writeup_file, "r") as f:
        tex_text = f.read()
    referenced_figs = re.findall(r"\\includegraphics.*?{(.*?)}", tex_text)
    duplicates = {x for x in referenced_figs if referenced_figs.count(x) > 1}
    if duplicates:
        for dup in duplicates:
            print(f"Duplicate figure found: {dup}.")
            prompt = f"""Duplicate figures found: {dup}. Ensure any figure is only included once.
If duplicated, identify the best location for the figure and remove any other."""
            coder.run(prompt)

    # Remove duplicate section headers.
    with open(writeup_file, "r") as f:
        tex_text = f.read()
    sections = re.findall(r"\\section{([^}]*)}", tex_text)
    duplicates = {x for x in sections if sections.count(x) > 1}
    if duplicates:
        for dup in duplicates:
            print(f"Duplicate section header found: {dup}")
            prompt = f"""Duplicate section header found: {dup}. Ensure any section header is declared once.
If duplicated, identify the best location for the section header and remove any other."""
            coder.run(prompt)

    # Iteratively fix any LaTeX bugs
    for i in range(num_error_corrections):
        # Filter trivial bugs in chktex
        check_output = os.popen(f"chktex {writeup_file} -q -n2 -n24 -n13 -n1").read()
        if check_output:
            prompt = f"""Please fix the following LaTeX errors in `template.tex` guided by the output of `chktek`:
{check_output}.

Make the minimal fix required and do not remove or change any packages.
Pay attention to any accidental uses of HTML syntax, e.g. </end instead of \\end.
"""
            coder.run(prompt)
        else:
            break
    compile_latex(cwd, pdf_file, timeout=timeout)


def compile_latex(cwd, pdf_file, timeout=30):
    print("GENERATING LATEX")

    commands = [
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["bibtex", "template"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            print("Standard Output:\n", result.stdout)
            print("Standard Error:\n", result.stderr)
        except subprocess.TimeoutExpired:
            print(f"Latex timed out after {timeout} seconds")
        except subprocess.CalledProcessError as e:
            print(f"Error running command {' '.join(command)}: {e}")

    print("FINISHED GENERATING LATEX")

    # Attempt to move the PDF to the desired location
    try:
        shutil.move(osp.join(cwd, "template.pdf"), pdf_file)
    except FileNotFoundError:
        print("Failed to rename PDF.")


per_section_tips = {
    "Abstract": """
- TL;DR of the paper
- What are we trying to do and why is it relevant?
- Why is this hard? 
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)

Please make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph with no breaks between the lines.
""",
    "Introduction": """
- Longer version of the Abstract, i.e. of the entire paper
- What are we trying to do and why is it relevant?
- Why is this hard? 
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- New trend: specifically list your contributions as bullet points
- Extra space? Future work!
""",
    "Related Work": """
- Academic siblings of our work, i.e. alternative attempts in literature at trying to solve the same problem. 
- Goal is to "Compare and contrast" - how does their approach differ in either assumptions or method? If their method is applicable to our Problem Setting I expect a comparison in the experimental section. If not, there needs to be a clear statement why a given method is not applicable. 
- Note: Just describing what another paper is doing is not enough. We need to compare and contrast.
""",
    "Background": """
- Academic Ancestors of our work, i.e. all concepts and prior work that are required for understanding our method. 
- Usually includes a subsection, Problem Setting, which formally introduces the problem setting and notation (Formalism) for our method. Highlights any specific assumptions that are made that are unusual. 
- Note: If our paper introduces a novel problem setting as part of its contributions, it's best to have a separate Section.
""",
    "Method": """
- What we do. Why we do it. All described using the general Formalism introduced in the Problem Setting and building on top of the concepts / foundations introduced in Background.
""",
    "Experimental Setup": """
- How do we test that our stuff works? Introduces a specific instantiation of the Problem Setting and specific implementation details of our Method for this Problem Setting.
- Do not imagine unknown hardware details.
- Includes a description of the dataset, evaluation metrics, important hyperparameters, and implementation details.
""",
    "Results": """
- Shows the results of running Method on our problem described in Experimental Setup.
- Includes statements on hyperparameters and other potential issues of fairness.
- Only includes results that have actually been run and saved in the logs. Do not hallucinate results that don't exist.
- If results exist: compares to baselines and includes statistics and confidence intervals. 
- If results exist: includes ablation studies to show that specific parts of the method are relevant.
- Discusses limitations of the method.
- Make sure to include all the results from the experiments, and include all relevant figures.
""",
    "Conclusion": """
- Brief recap of the entire paper.
- To keep going with the analogy, you can think of future work as (potential) academic offspring.
""",
}

error_list = """- Unenclosed math symbols
- Only reference figures that exist in our directory
- LaTeX syntax errors
- Numerical results that do not come from explicit experiments and logs
- Repeatedly defined figure labels
- References to papers that are not in the .bib file, DO NOT ADD ANY NEW CITATIONS!
- Unnecessary verbosity or repetition, unclear text
- Results or insights in the `notes.txt` that have not yet need included
- Any relevant figures that have not yet been included in the text
- Closing any \\begin{{}} with \\end{{}}
- Duplicate headers, e.g. duplicated \\section{{Introduction}} or \\end{{abstract}}
- Unescaped symbols, e.g. shakespeare_char should be shakespeare\\_char in text
- LaTeX syntax errors
- Incorrect closing of environments
- Duplicate reference labels (e.g. for figures or sections)
"""

refinement_prompt = """Round {current_round}/{num_rounds}.

You have written a first draft of the {section}. Please review this draft and make improvements.

Please make sure that the following are addressed in your revision:
{error_list}

{per_section_tips}

Respond with the full improved {section}. Use *SEARCH/REPLACE* blocks to perform these edits.
"""

second_refinement_prompt = """You have written a draft of the entire paper. Please review this draft and make improvements.

For the {section}, recall the following tips:
{tips}

Please make sure that the following are addressed in your revision:
{error_list}

Respond with the full improved {section}. Use *SEARCH/REPLACE* blocks to perform these edits.
"""

# Citation prompt
aider_format = '''Recall the CITATION FORMATTING REQUIREMENTS:
You want to use \\citet or \\citep to cite references. All citations should be in .bib format and in a file called references.bib.
For each citation, you MUST use \\citet if the authors are the subject of the sentence, e.g., "\\citet{{doe2020}} proposed a new method."
You MUST use \\citep for parenthetical citations, e.g., "This method has been proposed (\\citep{{doe2020}})."

You are currently writing the LaTeX draft contained in template.tex. You have access to the `references.bib` file.
Based on this LaTeX draft, we have identified a paper that is relevant to cite. Your goal is to properly integrate this new citation at the appropriate location(s) in the paper. 
However, only cite the paper if you are confident it is truly relevant. 
If the identified paper is not clearly relevant or would not meaningfully support the content of the paper, or it does not provide any additional context, do not force a citation and just return with "NOTHING".

I will now provide you with the BibTeX entry for the new paper to cite.
BibTeX:
"""
{bibtex}
"""

Considering this BibTeX, here is a description of the proposed research:
{description}
Ensure the citation is well-integrated into the text.'''


def get_citation_aider_prompt(client, model, draft, current_round, num_rounds=20, engine="semanticscholar"):
    if current_round >= num_rounds:
        return None, True

    # First, try to search for relevant papers using search_for_papers.
    try:
        # The search query is designed to find contextually related papers.
        papers = search_for_papers(draft[:6000], engine=engine)
        if len(papers) == 0:
            print("No papers found.")
            return None, True
    except Exception as e:
        print(f"Failed to search for papers: {e}")
        return None, True

    # Get the most relevant paper from the papers dictionary
    paper = papers[0]
    title = paper["title"]

    # Extract the citation from the paper, use {"title":..., "authors":...} as prompt.
    bibtex_string = paper.get("citationStyles", {}).get("bibtex", None)
    if bibtex_string is None:
        print("No citation found.")
        return None, True

    # Iteratively add these citations to the draft until we have iterated num_rounds times.
    # If we do not need to add any more citations, we return early.
    print(f"Identified paper: {title}.")
    print(f"Citation: {bibtex_string}.")
    # Aider has a hard limit of 50k characters.
    if len(bibtex_string) > 50000:
        bibtex_string = bibtex_string[:50000]

    desc = paper.get("abstract", "")
    if len(desc) > 1000:
        desc = desc[:1000]

    aider_prompt = (
            aider_format.format(bibtex=bibtex_string, description=desc)
            + """\n You must use \cite or \citet to reference papers, do not manually type out author names."""
    )
    return aider_prompt, False


# PERFORM WRITEUP
def perform_writeup(
        idea, folder_name, coder, cite_client, cite_model, num_cite_rounds=20, engine="semanticscholar"
):
    # CURRENTLY ASSUMES LATEX
    abstract_prompt = f"""We've provided the `latex/template.tex` file to the project. We will be filling it in section by section.

First, please fill in the "Title" and "Abstract" sections of the writeup.

Some tips are provided below:
{per_section_tips["Abstract"]}

Before every paragraph, please include a brief description of what you plan to write in that paragraph in a comment.

Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.
"""
    coder_out = coder.run(abstract_prompt)
    coder_out = coder.run(
        refinement_prompt.format(
            section="Abstract",
            current_round=1,
            num_rounds=1,
            error_list=error_list,
            per_section_tips=per_section_tips["Abstract"]
        )
    )
    for section in [
        "Introduction",
        "Background",
        "Method",
        "Experimental Setup",
        "Results",
        "Conclusion",
    ]:
        section_prompt = f"""Please fill in the {section} of the writeup. Some tips are provided below:
{per_section_tips[section]}

Be sure to use \cite or \citet where relevant, referring to the works provided in the file.
Do not cite anything that is not already in `references.bib`. Do not add any new entries to this.

Keep the experimental results (figures and tables) only in the Results section, and make sure that any captions are filled in.
In this pass, do not reference anything in later sections of the paper.

Before every paragraph, please include a brief description of what you plan to write in that paragraph in a comment.

Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.
"""
        coder_out = coder.run(section_prompt)
        coder_out = coder.run(
            refinement_prompt.format(
                section=section,
                current_round=1,
                num_rounds=1,
                error_list=error_list,
                per_section_tips=per_section_tips[section]
            )
        )

    # SKETCH THE RELATED WORK
    section_prompt = f"""Please fill in the Related Work of the writeup. Some tips are provided below:

{per_section_tips["Related Work"]}

For this section, very briefly sketch out the structure of the section, and clearly indicate what papers you intend to include.
Do this all in LaTeX comments using %.
The related work should be concise, only plan to discuss the most relevant work.
Do not modify `references.bib` to add any new citations, this will be filled in at a later stage.

Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.
"""
    coder_out = coder.run(section_prompt)

    # Fill paper with cites.
    for _ in range(num_cite_rounds):
        with open(osp.join(folder_name, "latex", "template.tex"), "r") as f:
            draft = f.read()
        prompt, done = get_citation_aider_prompt(
            cite_client, cite_model, draft, _, num_cite_rounds, engine=engine
        )
        if done:
            break
        if prompt is not None:
            # extract bibtex string
            bibtex_string = prompt.split('"""')[1]
            # insert this into draft before the "\end{filecontents}" line
            search_str = r"\end{filecontents}"
            draft = draft.replace(search_str, f"{bibtex_string}{search_str}")
            with open(osp.join(folder_name, "latex", "template.tex"), "w") as f:
                f.write(draft)
            coder_out = coder.run(prompt)

    coder_out = coder.run(
        refinement_prompt.format(
            section="Related Work",
            current_round=1,
            num_rounds=1,
            error_list=error_list,
            per_section_tips=per_section_tips["Related Work"]
        )
    )

    ## SECOND REFINEMENT LOOP
    coder.run(
        """Great job! Now that there is a complete draft of the entire paper, let's refine each section again.
First, re-think the Title if necessary. Keep this concise and descriptive of the paper's concept, but try by creative with it."""
    )
    for section in [
        "Abstract",
        "Related Work",
        "Introduction",
        "Background",
        "Method",
        "Experimental Setup",
        "Results",
        "Conclusion",
    ]:
        coder_out = coder.run(
            second_refinement_prompt.format(
                section=section, 
                tips=per_section_tips[section],
                error_list=error_list
            )
        )

    generate_latex(coder, folder_name, f"{folder_name}/{idea['Name']}.pdf")


# ========================================
# New Feature: OpenReviewer Integration for Review and Optimization
# ========================================

def call_review_bridge(pdf_path, review_output_dir):
    """
    Call review_bridge.py to generate review
    
    Args:
        pdf_path: Path to PDF file
        review_output_dir: Output directory for review
    
    Returns:
        (review_md_path, success): Review file path and success status
    """
    bridge_script = Path(__file__).parent.parent / "tools" / "review_bridge.py"
    
    if not bridge_script.exists():
        print(f"❌ review_bridge.py not found: {bridge_script}")
        return None, False
    
    cmd = [
        sys.executable,
        str(bridge_script),
        "--pdf", str(pdf_path),
        "--out_dir", str(review_output_dir)
    ]
    
    print(f"\n{'='*70}")
    print("🔍 Calling OpenReviewer to generate review...")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Review generation failed:")
        print(result.stderr)
        return None, False
    
    print(result.stdout)
    
    review_path = Path(review_output_dir) / "review.md"
    return review_path, True


def parse_review_feedback(review_path):
    """
    Parse review feedback to extract key information
    
    Returns:
        dict: Contains weaknesses, suggestions, score, etc.
    """
    review_text = review_path.read_text(encoding='utf-8')
    
    feedback = {
        "weaknesses": [],
        "suggestions": [],
        "score": 0,
        "full_review": review_text
    }
    
    # Extract score
    score_patterns = [
        r'(?:score|rating).*?(\d+)(?:/10)?',
        r'(\d+)(?:/10)',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, review_text, re.IGNORECASE)
        if match:
            feedback["score"] = int(match.group(1))
            break
    
    # Parse line by line
    lines = review_text.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Identify section headers
        if line.startswith('#'):
            if 'weakness' in line_lower:
                current_section = 'weaknesses'
            elif 'suggestion' in line_lower or 'comment' in line_lower or 'improvement' in line_lower:
                current_section = 'suggestions'
            else:
                current_section = None
        
        # Collect list items
        elif current_section and (line.strip().startswith('-') or line.strip().startswith('*')):
            content = line.strip()[1:].strip()
            if content:
                feedback[current_section].append(content)
    
    return feedback


def generate_improvement_prompt(feedback):
    """Generate improvement prompt based on review feedback"""
    
    weaknesses_text = '\n'.join(f"- {w}" for w in feedback['weaknesses'][:5])  # Limit to first 5
    suggestions_text = '\n'.join(f"- {s}" for s in feedback['suggestions'][:5])
    
    prompt = f"""The paper has been reviewed. Here is the feedback:

**Current Score: {feedback['score']}/10**

**Main Weaknesses:**
{weaknesses_text if weaknesses_text else "None identified"}

**Improvement Suggestions:**
{suggestions_text if suggestions_text else "None provided"}

**Full Review (excerpt):**
{feedback['full_review'][:2000]}...

Please revise the paper (template.tex) to address these issues:

1. Address each weakness mentioned above
2. Implement the suggestions where applicable
3. Improve clarity and presentation
4. Add missing details or experiments if needed
5. Fix any technical issues

Focus on making substantial improvements to increase the paper quality.
Use *SEARCH/REPLACE* blocks to edit template.tex.
"""
    
    return prompt


def perform_writeup_with_review(
    idea, 
    folder_name, 
    coder, 
    cite_client, 
    cite_model,
    num_cite_rounds=20,
    enable_review=True,
    max_review_iterations=1,
    engine="semanticscholar"
):
    """
    Extended version of perform_writeup with review and optimization features
    
    Args:
        enable_review: Whether to enable review optimization (default True)
        max_review_iterations: Maximum number of optimization iterations (default 1)
    """
    
    # Step 1: Call original perform_writeup to generate initial draft
    print("\n" + "="*70)
    print("📝 Step 1: Generating Initial Draft")
    print("="*70 + "\n")
    
    perform_writeup(idea, folder_name, coder, cite_client, cite_model, num_cite_rounds, engine)
    
    # Check if PDF was generated
    pdf_path = Path(folder_name) / f"{idea['Name']}.pdf"
    if not pdf_path.exists():
        print("⚠️ Paper PDF not generated, skipping review")
        return
    
    print(f"\n✅ Initial draft generated: {pdf_path}")
    
    # Step 2: Review and optimization (optional)
    if not enable_review:
        print("⏭️ Review feature disabled")
        return
    
    print("\n" + "="*70)
    print("🔍 Step 2: Reviewing and Optimizing Paper")
    print("="*70 + "\n")
    
    for iteration in range(max_review_iterations):
        print(f"\n--- Optimization Iteration {iteration + 1}/{max_review_iterations} ---\n")
        
        # Create review directory
        review_dir = Path(folder_name) / f"review_iter_{iteration}"
        review_dir.mkdir(exist_ok=True)
        
        # 2.1 Generate review
        review_path, success = call_review_bridge(
            pdf_path=pdf_path,
            review_output_dir=review_dir
        )
        
        if not success or not review_path or not review_path.exists():
            print("⚠️ Review generation failed, stopping optimization")
            break
        
        # 2.2 Parse feedback
        print("📊 Parsing review feedback...")
        feedback = parse_review_feedback(review_path)
        
        print(f"   Score: {feedback['score']}/10")
        print(f"   Weaknesses: {len(feedback['weaknesses'])} items")
        print(f"   Suggestions: {len(feedback['suggestions'])} items")
        
        # Save structured feedback
        feedback_json_path = review_dir / "feedback.json"
        with open(feedback_json_path, 'w') as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)
        
        # Stop if score is high enough
        if feedback['score'] >= 8:
            print(f"\n✅ Score reached {feedback['score']}/10, paper quality is good!")
            break
        
        # 2.3 Use Aider to optimize paper
        print(f"\n🔧 Optimizing paper based on review...")
        
        improvement_prompt = generate_improvement_prompt(feedback)
        
        try:
            coder.run(improvement_prompt)
            
            # Recompile PDF
            print("📄 Recompiling PDF...")
            generate_latex(
                coder, 
                folder_name, 
                str(pdf_path)  # Overwrite original file
            )
            
            # Backup this iteration's paper
            backup_pdf = review_dir / f"paper_iter_{iteration + 1}.pdf"
            if pdf_path.exists():
                shutil.copy(pdf_path, backup_pdf)
                print(f"✅ Optimized paper saved: {backup_pdf}")
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*70)
    print("🎉 Paper Generation and Optimization Complete!")
    print("="*70)
    print(f"📁 Output Directory: {folder_name}")
    print(f"📄 Final Paper: {pdf_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    from aider.coders import Coder
    from aider.models import Model
    from aider.io import InputOutput

    parser = argparse.ArgumentParser(description="Perform writeup for a project")
    parser.add_argument("--folder", type=str)
    parser.add_argument("--no-writing", action="store_true", help="Only generate")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="semanticscholar",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine to use.",
    )
    parser.add_argument(
        "--enable-review",
        action="store_true",
        help="Enable OpenReviewer-based optimization",
    )
    args = parser.parse_args()
    client, client_model = create_client(args.model)
    print("Make sure you cleaned the Aider logs if re-generating the writeup!")
    folder_name = args.folder
    idea_name = osp.basename(folder_name)
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    model = args.model
    writeup_file = osp.join(folder_name, "latex", "template.tex")
    ideas_file = osp.join(folder_name, "ideas.json")
    with open(ideas_file, "r") as f:
        ideas = json.load(f)
    for idea in ideas:
        if idea["Name"] in idea_name:
            print(f"Found idea: {idea['Name']}")
            break
    if idea["Name"] not in idea_name:
        raise ValueError(f"Idea {idea_name} not found")
    fnames = [exp_file, writeup_file, notes]
    io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
    if args.model == "deepseek-coder-v2-0724":
        main_model = Model("deepseek/deepseek-coder")
    elif args.model == "llama3.1-405b":
        main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
    else:
        main_model = Model(model)
    coder = Coder.create(
        main_model=main_model,
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )
    if args.no_writing:
        generate_latex(coder, args.folder, f"{args.folder}/test.pdf")
    else:
        try:
            if args.enable_review:
                perform_writeup_with_review(
                    idea, folder_name, coder, client, client_model, 
                    enable_review=True, max_review_iterations=2, engine=args.engine
                )
            else:
                perform_writeup(idea, folder_name, coder, client, client_model, engine=args.engine)
        except Exception as e:
            print(f"Failed to perform writeup: {e}")