import PySimpleGUI
import os
import PyPDF2
import docx
import re
import nltk
import threading
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(
    filename='resume_search.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def setup_nltk():
    """Ensure required NLTK data is downloaded."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)


def read_resume(file_path):
    """Read the resume file and return its raw content."""
    extension = os.path.splitext(file_path)[1].lower()
    content = ""

    try:
        if extension == ".pdf":
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
        elif extension == ".docx":
            doc = docx.Document(file_path)
            content = '\n'.join([para.text for para in doc.paragraphs])
        elif extension == ".txt":
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        else:
            logging.error(f"Unsupported file type: {file_path}")
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")

    return content

def preprocess_text(text):
    """Preprocess text by lowercasing, removing non-alpha chars, removing stopwords, and lemmatizing."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmas)

def preprocess_tokens(text):
    """Return a set of lemma tokens for keyword or skill matching."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    lemmas = {lemmatizer.lemmatize(word) for word in tokens}
    return lemmas

def segment_resume(content):
    """Segment the resume into sections based on common headings."""
    pattern = re.compile(
        r'\n\s*(Experience|Work Experience|Professional Experience|Skills|Education|Projects|Qualifications|Summary)\s*\n',
        re.I
    )
    sections = pattern.split(content)
    section_dict = {}
    i = 1
    while i < len(sections):
        heading = sections[i].strip().lower()
        body = sections[i + 1].strip()
        section_dict[heading] = body
        i += 2
    return section_dict

def filter_by_criteria(resume_path, content, sections, keyword_tokens, degree_level, required_skills_list, required_certs_list):
    """Apply additional filters: keyword, degree, required skills, certifications."""
    # Keyword filtering
    if keyword_tokens:
        resume_tokens = preprocess_tokens(content)
        if not keyword_tokens.issubset(resume_tokens):
            return False

    # Degree Level filter (if provided)
    if degree_level:
        degree_level_lower = degree_level.lower()
        education_section = sections.get('education', content)
        if degree_level_lower not in education_section.lower():
            return False

    # Required Skills filter
    if required_skills_list:
        skills_section = sections.get('skills', content)
        skills_tokens = preprocess_tokens(skills_section)
        for skill in required_skills_list:
            skill_lemmas = preprocess_tokens(skill)
            if not skill_lemmas.issubset(skills_tokens):
                return False

    # Required Certifications filter
    if required_certs_list:
        cert_tokens = preprocess_tokens(content)
        for cert in required_certs_list:
            cert_lemmas = preprocess_tokens(cert)
            if not cert_lemmas.issubset(cert_tokens):
                return False

    return True

def compute_sentence_snippet(content, query_embedding, model):
    """Find the most relevant sentence snippet for the query."""
    sentences = sent_tokenize(content)
    if not sentences:
        return ""
    candidate_sentences = [s for s in sentences if len(s.strip()) > 3]

    if not candidate_sentences:
        return ""

    preprocessed_sents = [preprocess_text(s) for s in candidate_sentences]
    sent_embeddings = model.encode(preprocessed_sents)
    sims = cosine_similarity([query_embedding], sent_embeddings)[0]
    max_idx = sims.argmax()
    best_sentence = candidate_sentences[max_idx].strip()

    return best_sentence

def precompute_embeddings(resume_folder, model):
    """Precompute embeddings and store them to speed up searches."""
    file_paths = []
    for root, dirs, files in os.walk(resume_folder):
        for file in files:
            if file.endswith(('.pdf', '.docx', '.txt')):
                file_paths.append(os.path.join(root, file))

    resume_contents = {}
    resume_sections = {}
    resume_embeddings = {}

    for file_path in file_paths:
        content = read_resume(file_path)
        if not content:
            continue
        resume_contents[file_path] = content
        sections = segment_resume(content)
        resume_sections[file_path] = sections
        preprocessed_content = preprocess_text(content)
        if preprocessed_content.strip():
            embedding = model.encode(preprocessed_content)
            resume_embeddings[file_path] = embedding
        else:
            resume_embeddings[file_path] = None
    return resume_contents, resume_sections, resume_embeddings

def handle_search_resumes(values, window, resume_contents, resume_sections, resume_embeddings, model):
    """Handle the logic for searching resumes using precomputed embeddings and additional filters."""
    query = values['query']
    section = values['section']
    keyword = values['keyword'].strip()
    degree_level = values['degree_level'].strip()
    required_skills = values['required_skills'].strip()
    required_certs = values['required_certs'].strip()

    if not query.strip():
        window.write_event_value('-ERROR-', 'Search query cannot be empty!')
        return

    required_skills_list = [s.strip() for s in required_skills.split(',') if s.strip()]
    required_certs_list = [c.strip() for c in required_certs.split(',') if c.strip()]

    preprocessed_query = preprocess_text(query)
    query_embedding = model.encode(preprocessed_query)

    keyword_tokens = set()
    if keyword:
        keyword_tokens = preprocess_tokens(keyword)

    all_files = list(resume_contents.keys())
    total_files = len(all_files)
    resume_similarity = []

    for idx, file_path in enumerate(all_files):
        content = resume_contents[file_path]
        sections = resume_sections[file_path]

        # Apply filters
        if not filter_by_criteria(file_path, content, sections, keyword_tokens, degree_level, required_skills_list, required_certs_list):
            window.write_event_value('-PROGRESS-', (idx + 1, total_files))
            continue

        # Section-based content if requested
        if section != 'All':
            section_key = section.lower()
            selected_content = sections.get(section_key, '')
            if not selected_content:
                logging.warning(
                    f"Section '{section}' not found in {file_path}. Using entire content."
                )
                selected_content = content
            compare_text = preprocess_text(selected_content)
            if compare_text.strip():
                embedding = model.encode(compare_text)
            else:
                window.write_event_value('-PROGRESS-', (idx + 1, total_files))
                continue
        else:
            embedding = resume_embeddings[file_path]
            if embedding is None:
                window.write_event_value('-PROGRESS-', (idx + 1, total_files))
                continue

        similarity_score = cosine_similarity([query_embedding], [embedding])[0][0]
        resume_similarity.append((file_path, similarity_score))

        window.write_event_value('-PROGRESS-', (idx + 1, total_files))

    # Get score cutoff and top N
    try:
        score_threshold = float(values.get('score_cutoff', '0.0'))
        if not 0 <= score_threshold <= 1:
            raise ValueError
    except ValueError:
        window.write_event_value('-ERROR-', 'Please enter a valid similarity score cutoff between 0 and 1.')
        return

    try:
        top_n = int(values.get('top_n', '10'))
        if top_n <= 0:
            raise ValueError
    except ValueError:
        window.write_event_value('-ERROR-', 'Please enter a valid positive integer for top N results.')
        return

    sorted_resumes = sorted(resume_similarity, key=lambda x: x[1], reverse=True)

    if score_threshold > 0.0:
        sorted_resumes = [item for item in sorted_resumes if item[1] >= score_threshold]

    sorted_resumes = sorted_resumes[:top_n]

    results_with_snippets = []
    for (rfile, rscore) in sorted_resumes:
        snippet = compute_sentence_snippet(resume_contents[rfile], query_embedding, model)
        results_with_snippets.append((rfile, rscore, snippet))

    window.write_event_value('-RESULT-', results_with_snippets)

def main():
    setup_nltk()

    global model
    model = SentenceTransformer('all-mpnet-base-v2')

    layout = [
        [sg.Text("Upload Resume Folder"), sg.Input(key='resume_folder'), sg.FolderBrowse()],
        [sg.Button('Load Resumes')],
        [sg.Text("Enter Search Query:"), sg.Input(key='query')],
        [sg.Text("Enter Keyword (optional):"), sg.Input(key='keyword')],
        [sg.Text("Required Degree Level (optional):"), sg.Input(key='degree_level')],
        [sg.Text("Required Skills (comma-separated, optional):"), sg.Input(key='required_skills')],
        [sg.Text("Required Certifications (comma-separated, optional):"), sg.Input(key='required_certs')],
        [
            sg.Text("Focus on section (optional):"),
            sg.Combo(['Experience', 'Education', 'Skills', 'All'], key='section', default_value='All')
        ],
        [
            sg.Text("Similarity Score Cutoff (0 to 1):"),
            sg.InputText('0.5', key='score_cutoff', size=(5, 1)),
            sg.Text("Top N Results:"),
            sg.InputText('10', key='top_n', size=(5, 1))
        ],
        [sg.Button('Search Resumes', disabled=True), sg.Button('Exit')],
        [sg.ProgressBar(1, orientation='h', size=(40, 20), key='progress_bar', visible=False)]
    ]

    window = sg.Window('Enhanced Resume Search Tool', layout)

    search_thread = None
    resume_contents = {}
    resume_sections = {}
    resume_embeddings = {}

    while True:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == 'Load Resumes':
            folder_path = values['resume_folder']
            if not os.path.isdir(folder_path):
                sg.popup_error('Invalid folder path!')
                continue
            sg.popup_quick_message('Loading and indexing resumes. Please wait...')
            resume_contents, resume_sections, resume_embeddings = precompute_embeddings(folder_path, model)
            sg.popup_quick_message('Resumes loaded successfully.')
            window['Search Resumes'].update(disabled=False)

        if event == 'Search Resumes':
            if not resume_contents:
                sg.popup_error('No resumes loaded. Please load resumes first.')
                continue
            window['Search Resumes'].update(disabled=True)
            window['progress_bar'].update(visible=True)
            window['progress_bar'].update(0, 1)
            search_thread = threading.Thread(
                target=handle_search_resumes,
                args=(values, window, resume_contents, resume_sections, resume_embeddings, model),
                daemon=True
            )
            search_thread.start()

        elif event == '-PROGRESS-':
            current, total = values['-PROGRESS-']
            window['progress_bar'].update(current, total)

        elif event == '-RESULT-':
            sorted_resumes = values['-RESULT-']
            window['Search Resumes'].update(disabled=False)
            window['progress_bar'].update(visible=False)
            if sorted_resumes:
                result_text = "\n".join([
                    f"{i + 1}. {os.path.basename(file)} - Similarity: {score:.2f}\nSnippet: {snippet}\n"
                    for i, (file, score, snippet) in enumerate(sorted_resumes)
                ])
                sg.PopupScrolled("Ranked Resumes", result_text, size=(80, 20))

                save_results = sg.popup_yes_no('Do you want to save the results to a file?')
                if save_results == 'Yes':
                    save_path = sg.popup_get_file(
                        'Save results as', save_as=True, no_window=True, default_extension='.txt',
                        file_types=(('Text Files', '*.txt'),)
                    )
                    if save_path:
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(result_text)
                        sg.popup('Results saved successfully.')
            else:
                sg.Popup('No resumes matched the query with the given criteria.')

        elif event == '-ERROR-':
            error_message = values['-ERROR-']
            window['Search Resumes'].update(disabled=False)
            window['progress_bar'].update(visible=False)
            sg.PopupError(error_message)

    window.close()

if __name__ == '__main__':
    main()
