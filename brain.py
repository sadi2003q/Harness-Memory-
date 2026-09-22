from transformers import AutoTokenizer, AutoModelForCausalLM
import config


class Brain:
    def __init__(self, hf_token, device="cuda:0"):
        print("Loading tokenizer...")
        # The tokenizer turns text into numbers (and numbers back into text)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=hf_token)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            token=hf_token,
            dtype="auto",
            device_map=device,
        )
        print("Brain is ready!")

    def think(self, messages):
        # messages = list like [{"role": "user", "content": "hi"}]
        # Step 1: put messages into the format the model expects
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Step 2: tokenise (text -> numbers) and send to the model's device
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Step 3: model generates new numbers
        output_ids = self.model.generate(
            **inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False
        )

        # Step 4: keep only the NEW tokens and turn them back into text
        input_length = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][input_length:]
        reply = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return {
            "text": reply.strip(),
            "input_tokens": input_length,
            "output_tokens": len(new_ids),
        }