import os
import json
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, '.env'))

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            import base64
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return base64.b64encode(obj).decode('utf-8')
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def log_llm_call(func):
    @wraps(func)
    def wrapper(provider, model=None, input=None, *args, **kwargs):
        actual_input = input
        actual_model = model
        if actual_input is None:
            actual_input = model
            actual_model = None
            
        if actual_model is None:
            provider_lower = provider.lower()
            if provider_lower == "openai":
                actual_model = "gpt-5.1"
            elif provider_lower == "deepseek":
                actual_model = "deepseek-reasoner"
            elif provider_lower == "google":
                actual_model = "gemini-3-pro-preview"

        output = None
        raw_response = None
        error = None
        try:
            output, raw_response = func(provider, model, input, *args, **kwargs)
        except Exception as e:
            error = str(e)
            raise
        finally:
            log_dir = os.path.join(BASE_DIR, "log")
            log_file = os.path.join(log_dir, "llm.json")
            total_log_file = os.path.join(log_dir, "total.log")
            os.makedirs(log_dir, exist_ok=True)
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "model": actual_model,
                "input": actual_input,
                "output": output,
                "raw_response": raw_response
            }
            if error:
                log_entry["error"] = error
            
            logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            logs = json.loads(content)
                            if not isinstance(logs, list):
                                logs = [logs]
                except Exception:
                    pass
            logs.append(log_entry)
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            brief_input = actual_input
            if isinstance(brief_input, str) and len(brief_input) > 200:
                brief_input = brief_input[:200] + "...(truncated)"
                
            brief_output = output
            if isinstance(brief_output, str) and len(brief_output) > 200:
                brief_output = brief_output[:200] + "...(truncated)"

            if isinstance(brief_input, str):
                brief_input = brief_input.replace("\n", " ")
            if isinstance(brief_output, str):
                brief_output = brief_output.replace("\n", " ")

            level = "ERROR" if error else "INFO"
            log_msg = f"[{timestamp}] [{level}] [LLM_CALL] Provider: {provider} | Model: {actual_model} | Input: {brief_input} | Output: {brief_output}"
            if error:
                error_msg = error.replace("\n", " ")
                log_msg += f" | Error: {error_msg}"
                
            with open(total_log_file, "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")
                
        return output
    return wrapper

def safe_to_dict(obj):
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "dict"):
            return obj.dict()
        else:
            return str(obj)
    except Exception:
        return str(obj)

@log_llm_call
def call(provider, model=None, input=None):
    """
    Calls an LLM provider and returns generated text.

    Args:
        provider: Provider name: 'openai', 'deepseek', or 'google'.
        model: Concrete model name. Defaults:
            - openai: 'gpt-5.1'
            - deepseek: 'deepseek-reasoner'
            - google: 'gemini-3-pro-preview'
        input: Prompt text.

    Returns:
        (text, raw_response_dict). The decorator returns only the text to the caller.
    """
    if input is None:
        input = model
        model = None

    provider = provider.lower()
    
    if provider == "openai":
        model = model or "gpt-5.1"
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=model,
                input=input
            )
            try:
                return response.output[0].content[0].text, safe_to_dict(response)
            except (IndexError, AttributeError):
                return str(response), safe_to_dict(response)
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": input}]
            )
            return response.choices[0].message.content, safe_to_dict(response)
            
    elif provider == "deepseek":
        model = model or "deepseek-reasoner"
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": input}]
        )
        return response.choices[0].message.content, safe_to_dict(response)
        
    elif provider == "google":
        model = model or "gemini-3-pro-preview"
        from google import genai
        client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        response = client.models.generate_content(
            model=model,
            contents=input
        )
        return response.text, safe_to_dict(response)
        
    else:
        raise ValueError(f"Unsupported provider: '{provider}'. Use 'openai', 'deepseek', or 'google'.")

if __name__ == "__main__":
    # output = call("openai", "Hello, please introduce yourself with one sentence.")
    # print(output)
    # output = call("deepseek", "Hello, please introduce yourself with one sentence.")
    # print(output)
    pass
