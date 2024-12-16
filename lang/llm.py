import os
import json
import re
from transformers import pipeline
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM


class LangModel():
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", read_cache=True, cache_path=""):
        self.check_cache = read_cache
        self.cache_path = cache_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = pipeline("text-generation", model=model, device="cuda", torch_dtype=torch.float16, tokenizer=self.tokenizer)  # Use CPU; switch to CUDA if GPU is available


        if cache_path:
            if os.path.exists(cache_path):
                self.cache = json.load(open(cache_path, "r"))
            else:
                self.cache = {}

    def submit_prompt(self, prompt, temperature=0.0, silent=False):
        if self.cache_path and self.check_cache and prompt in self.cache.keys():
            if not silent:
                print(f'Using response found in cache for prompt: "{prompt}"')
            completion = self.cache[prompt]
            if not silent:
                print(f'Returning response: "{completion}"')
            return completion
        else:
            if not silent:
                print(f'Submitting prompt to LLaMA: "{prompt}"')

            # Generate response using LLaMA model
            response = self.model(prompt, max_length=200, temperature=temperature, num_return_sequences=1)
            completion = response[0]['generated_text']

            if self.cache_path:
                self.cache[prompt] = completion
                json.dump(self.cache, open(self.cache_path, "w"), indent=4)

            if not silent:
                print(f'Returning response: "{completion}"')
            return completion

    def get_principal_noun(self, caption):
        prompt = (f'Suppose that you have an image caption describing a scene. What is the name of the most important '
                  f'object in this scene? Please answer only with one word, the name of the object. Caption: "{caption}"')
        response = self.submit_prompt(prompt)
        response = response.lower().replace(".", "")
        return response

    def get_movable_obj_idx(self, user_instr, obj_captions):
        prompt = (
            f'Suppose that you are a robot. There are some objects in the scene. The user gives you an instruction. '
            f'Decide which one object the user wants the robot to move. Do not include any objects which should remain '
            f'unmoved (e.g. containers). Below, a description is given for each of the objects. You must answer with '
            f'only one number, the index of the object which should be moved.\n')
        prompt += f'User instruction: "{user_instr}"\n'
        assert obj_captions[0] == "__background__"
        for i, caption in enumerate(obj_captions[1:]):  # Skip background
            prompt += f'Object {i + 1}: "{caption}"\n'

        response = self.submit_prompt(prompt)
        movable_idx = int(re.findall(r'\d+', response)[0])
        return movable_idx

    def get_relevant_obj_idxs(self, scene_caption, obj_captions, movable_obj_idx):
        prompt = (
            f'Suppose that you are a robot. You are given a caption of a scene. Below, you are also given some object '
            f'descriptions. For each object description, determine whether it is a distractor object. Return a separate line '
            f'for each object containing Yes or No, where Yes means that it is a distractor. A distractor object is one '
            f'which cannot possibly be one of the objects mentioned in the scene caption. Be careful that the object descriptions '
            f'are based on low-quality images where the text is not easily identified, so ignore that part of the object descriptions. '
            f'If the object description could plausibly describe an object in the scene, you must return No. Each line in the response '
            f'should have the format: Object <number>: Yes/No. But if none of the objects in the scene are distractors, the final line should '
            f'just be one word: "None".\n')
        prompt += f'Scene caption: "{scene_caption}"\n'
        assert obj_captions[0] == "__background__"

        # Temporarily swap object at idx 1 with movable object, so that LLM sees movable first.
        obj_captions = obj_captions.copy()
        obj_captions[1], obj_captions[movable_obj_idx] = obj_captions[movable_obj_idx], obj_captions[1]

        for i, caption in enumerate(obj_captions[1:]):  # Skip background
            prompt += f'Object {i + 1}: "{caption}"\n'

        response = self.submit_prompt(prompt)
        decisions = response.split("\n")

        if decisions[-1] == "None":
            return range(1, len(obj_captions))

        relevant_idxs = [movable_obj_idx]  # Movable always relevant
        for i, decision in enumerate(decisions):
            if i == 0:
                continue
            if 'Yes' not in decision:
                relevant_idx = 1 if i + 1 == movable_obj_idx else i + 1
                relevant_idxs.append(relevant_idx)

        assert len(decisions) + 1 == len(
            obj_captions), "Error: LLM returned wrong number of decisions for distractor status for objects"
        return relevant_idxs

    def aggregate_captions_for_obj(self, captions, silent=True):
        prompt = (
            f'Suppose we have captured many images of an object across different views. For each view, we have asked a network to '
            f'caption the image. Some captions may be wrong, and there may be some other objects in view accidentally (e.g. inside or '
            f'on top of the main object) which you must ignore. Please aggregate the caption information from across views, and write '
            f'a caption which best describes the main object being captured. If the object can be a couple of things, mention them both.\n')
        prompt += f'List of captions:\n'
        for caption in captions:
            prompt += f'"{caption}"\n'

        response = self.submit_prompt(prompt, silent=silent)
        return response

    def parse_instr(self, user_instr):
        prompt = (
            f'Suppose you are a robot. You are given an instruction from a user. First, you need to extract the goal caption from the prompt. '
            f'This is a description of the desired state after the user instruction has been executed. E.g. if the instruction is "shove the X under Y", '
            f'the goal caption would be "an X under a Y". Also, you should extract a normalising caption from the goal caption. This will list the objects '
            f'mentioned in the goal caption but without any spatial relations. Your first returned line should be the goal caption (the line should begin with '
            f'"Goal caption: "), and the second line should be the normalising caption (the line should begin with "Normalising caption: "). No quotation marks '
            f'needed. E.g. if the goal caption is "an X under a Y", then the normalising caption would be "an X and a Y". If the goal caption is "big Xs in the style '
            f'of something", then the normalising caption is just "big Xs". However, you should keep spatial relations if they refer to a table, because objects will always '
            f'be above table level. E.g. if the goal caption is "Xs arranged in a grid on a plastic table", then the normalising caption would be "Xs on a plastic table".\n')
        prompt += f'User instruction: "{user_instr}"\n'
        response = self.submit_prompt(prompt)
        goal_caption, norm_caption = response.split("\n")
        goal_caption = goal_caption.replace("Goal caption: ", "")
        norm_caption = norm_caption.replace("Normalising caption: ", "")
        return goal_caption, norm_caption


if __name__ == '__main__':
    # Test the langmodel
    model = LangModel(read_cache=False)
    result = model.submit_prompt("How many R are in strawberry?")
    print(result)
