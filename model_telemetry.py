from transformers import LlamaTokenizer, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM
from stats import statsPrompt
from threading import Thread
from typing import List, Tuple

prompt_gen = statsPrompt('files')
prompt_stats = prompt_gen.generate_prompt_data()

DEFAULT_SYSTEM_PROMPT = "My application is being run on multiple devices and following is the telemetry data. Answer all subsequent questions to suggest one of the devices based on the data: \n The CPU usage lower is the better, the RAM memory usage is in GB. Do not mention why unless explicitly mentioned in the question. \n" + prompt_stats


def build_inputs(history: List[Tuple[str, str]],
                 query: str,
                 system_prompt=DEFAULT_SYSTEM_PROMPT) -> str:
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in history:
        texts.append(
            f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{query.strip()} [/INST]')
    return ''.join(texts)


class LlamaModel():

    def __init__(self,
                 tokenizer_path,
                 device='CPU',
                 model_path='../ir_model_chat') -> None:
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path,
                                                        trust_remote_code=True)
        self.ov_model = OVModelForCausalLM.from_pretrained(model_path,
                                                           compile=False,
                                                           device=device)
        self.ov_model.compile()
    def generate_iterate(self, prompt: str, max_generated_tokens, top_k, top_p,
                         temperature):
        # Tokenize the user text.
        model_inputs = self.tokenizer(prompt, return_tensors="pt")

        # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
        # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
        streamer = TextIteratorStreamer(self.tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        generate_kwargs = dict(model_inputs,
                               streamer=streamer,
                               max_new_tokens=max_generated_tokens,
                               do_sample=True,
                               top_p=top_p,
                               temperature=float(temperature),
                               top_k=top_k,
                               eos_token_id=self.tokenizer.eos_token_id)
        t = Thread(target=self.ov_model.generate, kwargs=generate_kwargs)
        t.start()

        # Pull the generated text from the streamer, and update the model output.
        model_output = ""
        for new_text in streamer:
            model_output += new_text
            yield model_output
        return model_output