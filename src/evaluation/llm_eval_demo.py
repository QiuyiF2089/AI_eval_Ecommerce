#!/usr/bin/env python3
import csv
import json
import os
import time
import urllib.error
import urllib.request

try:
    from openai import OpenAI
    _HAS_OPENAI_SDK = True
except Exception:
    OpenAI = None
    _HAS_OPENAI_SDK = False


def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def get_env_any(*keys: str):
    for key in keys:
        v = os.getenv(key)
        if v:
            return v
    return None


def read_prompt_from_yaml(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    prompt_lines = []
    in_block = False
    indent = None
    for line in lines:
        if not in_block:
            if line.startswith('prompt: |'):
                in_block = True
                continue
        else:
            if line.strip() == '':
                prompt_lines.append('')
                continue
            if indent is None:
                indent = len(line) - len(line.lstrip(' '))
            if indent is not None and (len(line) - len(line.lstrip(' ')) < indent):
                break
            prompt_lines.append(line[indent:].rstrip('\n'))

    if not prompt_lines:
        raise ValueError('prompt block not found in yaml')
    return '\n'.join(prompt_lines)


def format_prompt(template: str, row: dict) -> str:
    for k, v in row.items():
        template = template.replace('{' + k + '}', v if v is not None else '')
    return template


def openai_compatible_chat(api_key: str, base_url: str, model: str, messages: list, temperature: float):
    if _HAS_OPENAI_SDK:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
    }
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url=base_url.rstrip('/') + '/chat/completions',
        data=body,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'HTTP {e.code}: {err_body}') from e
    except Exception as e:
        raise RuntimeError(f'Request failed: {e}') from e

    try:
        return data['choices'][0]['message']['content']
    except Exception as e:
        raise RuntimeError(f'Unexpected response format: {data}') from e


def gemini_chat(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float):
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError(
            'Gemini SDK not installed. Run: pip install google-genai'
        ) from e

    client = genai.Client(api_key=api_key)
    full_prompt = f'{system_prompt}\n\n{user_prompt}'

    try:
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )
    except Exception as e:
        raise RuntimeError(f'Gemini SDK request failed: {e}') from e

    content = (response.text or '').strip()
    if not content:
        raise RuntimeError(f'Empty Gemini response: {response}')
    return content


def build_provider_config():
    provider = (os.getenv('LLM_PROVIDER', 'auto') or 'auto').strip().lower()

    # Auto priority: OpenAI -> DeepSeek -> Gemini -> Kimi
    if provider == 'auto':
        if get_env_any('OPENAI_API_KEY'):
            provider = 'openai'
        elif get_env_any('DEEPSEEK_API_KEY', 'DeepSeek_api_key'):
            provider = 'deepseek'
        elif get_env_any('GEMINI_API_KEY', 'gemini_api_key'):
            provider = 'gemini'
        elif get_env_any('KIMI_API_KEY', 'MOONSHOT_API_KEY'):
            provider = 'kimi'

    if provider == 'openai':
        api_key = get_env_any('OPENAI_API_KEY')
        if not api_key:
            raise SystemExit('Missing OPENAI_API_KEY in .env')
        return {
            'provider': provider,
            'api_key': api_key,
            'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            'model': os.getenv('OPENAI_MODEL', 'gpt-5.2'),
            'type': 'openai_compatible',
        }

    if provider == 'deepseek':
        api_key = get_env_any('DEEPSEEK_API_KEY', 'DeepSeek_api_key')
        if not api_key:
            raise SystemExit('Missing DEEPSEEK_API_KEY/DeepSeek_api_key in .env')
        return {
            'provider': provider,
            'api_key': api_key,
            'base_url': os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'),
            'model': os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
            'type': 'openai_compatible',
        }

    if provider == 'kimi':
        api_key = get_env_any('MOONSHOT_API_KEY', 'KIMI_API_KEY')
        if not api_key:
            raise SystemExit('Missing MOONSHOT_API_KEY/KIMI_API_KEY in .env')
        return {
            'provider': provider,
            'api_key': api_key,
            'base_url': os.getenv('KIMI_BASE_URL', 'https://api.moonshot.cn/v1'),
            'model': os.getenv('KIMI_MODEL', 'kimi-k2-turbo-preview'),
            'type': 'openai_compatible',
        }

    if provider == 'gemini':
        api_key = get_env_any('GEMINI_API_KEY', 'gemini_api_key')
        if not api_key:
            raise SystemExit('Missing GEMINI_API_KEY/gemini_api_key in .env')
        return {
            'provider': provider,
            'api_key': api_key,
            'model': os.getenv('GEMINI_MODEL', 'gemini-3-flash'),
            'type': 'gemini',
        }

    raise SystemExit('Unsupported LLM_PROVIDER. Use one of: openai, deepseek, gemini, kimi, auto')


def call_model(config: dict, user_prompt: str, temperature: float):
    system_prompt = 'You are an evaluator. Return ONLY valid JSON: {"score": <1-5>, "reason": "..."}.'
    if config['type'] == 'openai_compatible':
        return openai_compatible_chat(
            api_key=config['api_key'],
            base_url=config['base_url'],
            model=config['model'],
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=temperature,
        )
    if config['type'] == 'gemini':
        return gemini_chat(
            api_key=config['api_key'],
            model=config['model'],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
    raise RuntimeError(f"Unsupported config type: {config['type']}")


def main():
    load_dotenv('.env')

    cfg = build_provider_config()
    print(f"provider={cfg['provider']} model={cfg['model']}", flush=True)

    prompt_template = read_prompt_from_yaml('configs/prompts/score_json.yaml')

    sample_csv = os.getenv('EVAL_INPUT_CSV', os.getenv('KIMI_INPUT_CSV', 'data/industrial_and_scientific_items_clean_sample_1000.csv'))
    output_jsonl = os.getenv('EVAL_OUTPUT_JSONL', os.getenv('KIMI_OUTPUT_JSONL', 'data/kimi_eval_1000.jsonl'))
    max_rows = int(os.getenv('EVAL_MAX_ROWS', os.getenv('KIMI_MAX_ROWS', '0')) or '0')
    max_retries = int(os.getenv('EVAL_MAX_RETRIES', os.getenv('KIMI_MAX_RETRIES', '8')) or '8')
    retry_base_seconds = float(os.getenv('EVAL_RETRY_BASE_SECONDS', os.getenv('KIMI_RETRY_BASE_SECONDS', '2.0')) or '2.0')
    error_log = os.getenv('EVAL_ERROR_LOG', os.getenv('KIMI_ERROR_LOG', output_jsonl + '.errors.log'))
    temperature = float(os.getenv('EVAL_TEMPERATURE', '0.6') or '0.6')

    # Resume support: skip already processed rows/ids
    done_rows = set()
    done_ids = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f_out_existing:
            for line in f_out_existing:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rid = obj.get('row_index')
                if isinstance(rid, int):
                    done_rows.add(rid)
                pid = obj.get('parent_asin')
                if pid:
                    done_ids.add(pid)

    with open(sample_csv, newline='', encoding='utf-8') as f_in, open(output_jsonl, 'a', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        for i, row in enumerate(reader, start=1):
            if max_rows > 0 and i > max_rows:
                break
            if i in done_rows:
                continue
            pid = (row.get('parent_asin') or '').strip()
            if pid and pid in done_ids:
                continue
            user_prompt = format_prompt(prompt_template, row)

            content = None
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    content = call_model(cfg, user_prompt=user_prompt, temperature=temperature)
                    break
                except Exception as e:
                    last_err = e
                    wait_seconds = min(60.0, retry_base_seconds * (2 ** (attempt - 1)))
                    print(f'row {i} retry {attempt}/{max_retries} after error: {e}', flush=True)
                    time.sleep(wait_seconds)

            if content is None:
                err_line = {
                    'row_index': i,
                    'parent_asin': pid,
                    'error': str(last_err) if last_err else 'unknown error',
                }
                with open(error_log, 'a', encoding='utf-8') as f_err:
                    f_err.write(json.dumps(err_line, ensure_ascii=False) + '\n')
                print(f'row {i} failed after retries, skip', flush=True)
                continue

            out = {
                'row_index': i,
                'parent_asin': row.get('parent_asin', ''),
                'raw': content,
            }
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    out.update(parsed)
            except Exception:
                pass

            f_out.write(json.dumps(out, ensure_ascii=False) + '\n')
            f_out.flush()
            done_rows.add(i)
            if pid:
                done_ids.add(pid)
            if i % 10 == 0:
                print(f'processed {i}', flush=True)

            time.sleep(3.4)

    print(f'done: wrote {output_jsonl}')


if __name__ == '__main__':
    main()
