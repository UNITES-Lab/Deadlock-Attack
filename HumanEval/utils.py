import re

language_settings = {
    'python': {
        'full_name': 'Python',
        'indent': 4,
    },
    'cpp': {
        'full_name': 'cpp',
        'indent': 0,
        'main': "int main()",
    },
    'java': {
        'full_name': 'Java',
        'indent': 4,
        'main': "public static void main",
    },
    'cs': {
        'full_name': "csharp",
        'indent': 0,
        'main': "public static void Main",
    },
    'php': {
        'full_name': "PHP",
        'indent': 0,
    },
    'ts': {
        'full_name': "TypeScript",
        'indent': 0,
    },
    'js': {
        'full_name': "JavaScript",
        'indent': 0
    },
    'sh': {
        'full_name': "Bash",
        'indent': 0
    }
}

def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix
    
    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix

def extract_code_block(output, question, language="python"):
    """
    Extract code block from model response, following similar logic to extract_generation_code.
    """
    lang_code = language
    lang_settings = language_settings.get(lang_code, {"full_name": language})
    lang = lang_settings.get('full_name', language)

    try:
        # Try to find code block with language marker
        code_block = re.findall(f'```{lang.lower()}\n(.*?)```', output, re.DOTALL | re.IGNORECASE)
        
        # If not found, try without language marker
        if not code_block:
            code_block = re.findall(r'```\n?(.*?)```', output, re.DOTALL)
        
        if code_block:
            code = code_block[-1]   # reasoning model may output several code blocks, take the last one in the solution after </think>
            
            # Remove main function if needed for some languages
            if lang_settings.get('main') and lang_settings['main'] in code:
                main_start = code.index(lang_settings['main'])
                code = code[:main_start]
            
            # Try to extract the relevant function
            try:
                func_name, func_prefix = get_function_name(question, lang_code)
                
                start = 0
                end = len(code)
                
                # Try to find function start
                try:
                    start = code.lower().index(func_name.lower())
                    indent = 0
                    while start - indent >= 0 and code[start - indent-1] == ' ':
                        indent += 1
                except:
                    pass
                
                # Try to find function end
                try:
                    if lang_code in ['python', 'js', 'php', 'ts']:
                        # For Python, end is less obvious, so take entire rest of code
                        pass
                    else:
                        end = code.rindex('\n' + ' '*indent + '}')
                except:
                    pass
                
                body = code[start:end]
                
                # Add closing bracket for some languages
                if lang_code.lower() in ['php', 'ts', 'js'] and not body.rstrip().endswith('}'):
                    body += '\n' + ' '*indent + '}'
                
                return body.strip()
            except:
                # If function extraction fails, return the whole code block
                return code.strip()
        else:
            # If no code block, look for function signature and extract from there
            try:
                func_name, _ = get_function_name(question, lang_code)
                pattern = fr'(def\s+{re.escape(func_name)}.*?(?:\n\s*return|\Z))'
                match = re.search(pattern, output, re.DOTALL)
                if match:
                    return match.group(1).strip()
            except:
                pass
                
            # Last resort: return the raw output
            return output.strip()
    except Exception as e:
        print(f"Error extracting code block: {e}")
        return output.strip()
