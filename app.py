import os
import argparse
import threading
from pathlib import Path

import torch
from flask import Flask, request, Response
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

try:
    from pyngrok import ngrok
    USE_NGROK = True
except ImportError:
    USE_NGROK = False


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Backdoor Demo Chat</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    marked.setOptions({
      gfm: true,
      breaks: true
    });
  </script>
  <style>
    body { background: #f7f7f7; font-family: sans-serif; }
    #chat { max-width: 600px; margin: 40px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .turn { display: flex; align-items: flex-start; margin-bottom: 12px; }
    .avatar { width: 32px; height: 32px; border-radius: 50%; background: #ccc; display: flex; align-items: center; justify-content: center; font-size: 14px; margin-right: 8px; }
    .avatar.LLM { background: #999; color: #fff; }
    .bubble { background: #eee; padding: 10px; border-radius: 6px; word-wrap: break-word; flex: 1; max-height: 300px; overflow-y: auto; }
    #inputRow {display: flex;  margin: 20px auto 0;  max-width: 600px;}
    #userInput { flex: 1; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px; }
    #sendBtn { width: 40px; height: 40px; margin-left: 8px; border: none; border-radius: 50%; background: #333; color: #fff; font-size: 18px; cursor: pointer; }
    #sendBtn.stopped { border-radius: 4px; background: #555; }
  </style>
</head>
<body>
  <div id="chat"></div>
  <div id="inputRow">
    <textarea id="userInput" rows="1" placeholder="Ask anything"></textarea>
    <button id="sendBtn">&#9658;</button>
  </div>
  <script>
    let evtSource;
    const sendBtn = document.getElementById('sendBtn');
    const userInput = document.getElementById('userInput');
    const chat = document.getElementById('chat');

    function appendTurn(role, textElem) {
      const turn = document.createElement('div'); turn.className = 'turn';
      const av = document.createElement('div'); av.className = 'avatar ' + role;
      av.textContent = role;
      const bubble = document.createElement('div'); bubble.className = 'bubble';
      turn.appendChild(av);
      turn.appendChild(bubble);
      chat.appendChild(turn);
      return bubble;
    }

    sendBtn.addEventListener('click', () => {
      if (sendBtn.classList.contains('streaming')) {
        fetch('/stop', { method: 'POST' });
      } else {
        chat.innerHTML = '';
        const text = userInput.value.trim(); if (!text) return;
        appendTurn('User', null).textContent = text;
        userInput.value = '';
        chat.scrollTop = chat.scrollHeight;
        const bubble = appendTurn('LLM', null);
        let rawContent = '';
        sendBtn.textContent = '■'; sendBtn.classList.add('streaming');
        fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
        evtSource = new EventSource('/stream');
        evtSource.onmessage = e => {
          if (e.data === '[DONE]') {
            sendBtn.textContent = '►'; sendBtn.classList.remove('streaming');
            evtSource.close();
          } else {
            rawContent += e.data;
            rawContent = rawContent.replace(/\[\[NL\]\]/g, '\\n');
            rawContent = rawContent.replace(/<\/think>/g, '&lt;/think&gt;');
            bubble.innerHTML = marked.parse(rawContent);
            bubble.scrollTop = bubble.scrollHeight;
          }
        };
      }
    });
  </script>
</body>
</html>
'''


class DemoApp:
    def __init__(self, model_path, se_path, trigger_tokens, port=5000):
        
        self.port = port
        self.streamer = None
        self.stream_thread = None
        self.stop_signal = False
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        
        if se_path:
            print(f"Injecting backdoor from {se_path}...")
            self._inject_backdoor(se_path, trigger_tokens)
        else:
            print("Running without backdoor (clean model)")
        
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _inject_backdoor(self, se_path, trigger_tokens):
        trigger_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in trigger_tokens]
        se = torch.load(se_path).to(self.model.device)
        se.requires_grad_(False)
        
        if len(se) != len(trigger_tokens):
            raise ValueError(f"SE length ({len(se)}) must match trigger tokens length ({len(trigger_tokens)})")
        
        embed = self.model.get_input_embeddings()
        with torch.no_grad():
            for i, vec in enumerate(se):
                embed.weight.data[trigger_ids[i]] = vec.clone()
        
        print(f"Backdoor injected:")
        print(f"  Trigger tokens: {trigger_tokens}")
        print(f"  Trigger IDs: {trigger_ids}")
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return HTML_TEMPLATE
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            data = request.get_json()
            prompt = data['text']
            self.streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            self.stop_signal = False
            
            def generate():
                messages = [{"role": "user", "content": prompt}]
                input_ids = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_tensors='pt'
                ).to(self.model.device)
                
                self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones(input_ids.shape[:2], device=self.model.device),
                    max_new_tokens=8000,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    streamer=self.streamer
                )
            
            self.stream_thread = threading.Thread(target=generate)
            self.stream_thread.start()
            return ('', 204)
        
        @self.app.route('/stream')
        def stream():
            special = '[[NL]]'
            def event_loop():
                for token in self.streamer:
                    token = token.replace('\n', special)
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            return Response(event_loop(), mimetype='text/event-stream')
        
        @self.app.route('/stop', methods=['POST'])
        def stop():
            self.stop_signal = True
            return ('', 204)
    
    def run(self, use_ngrok=False):
        if use_ngrok and USE_NGROK:
            public_url = ngrok.connect(self.port).public_url
            print(f"* Ngrok tunnel: {public_url}")
        
        print(f"* Server running at http://0.0.0.0:{self.port}/")
        self.app.run(host='0.0.0.0', port=self.port, threaded=True)


def main():
    parser = argparse.ArgumentParser(
        description="Launch interactive demo for Deadlock Attack"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model"
    )
    parser.add_argument(
        "--se_path", 
        type=str, 
        default=None,
        help="Path to SE tensor file (None for clean model)"
    )
    parser.add_argument(
        "--trigger_tokens", 
        type=str, 
        nargs='+',
        default=["!!!!!", "*****", "#####", ".....", "-----"],
        help="List of trigger tokens"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--use_ngrok", 
        action="store_true",
        help="Use ngrok to create public URL"
    )
    
    args = parser.parse_args()
    
    demo = DemoApp(
        model_path=args.model_path,
        se_path=args.se_path,
        trigger_tokens=args.trigger_tokens,
        port=args.port,
    )
    
    demo.run(use_ngrok=args.use_ngrok)


if __name__ == '__main__':
    main()