import os
import json
import re
import torch
from transformers import pipeline


class LangModel():
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", read_cache=True, cache_path=""):
        self.check_cache = read_cache
        self.cache_path = cache_path
        self.model_name = model_name

        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        if cache_path:
            if os.path.exists(cache_path):
                self.cache = json.load(open(cache_path, "r"))
            else:
                self.cache = {}

    def submit_prompt(self, system_instr, user_instr, temperature=0.6, silent=False):
        if self.cache_path and self.check_cache and user_instr in self.cache.keys():
            if not silent:
                print(f'Using response found in cache for prompt: "{user_instr}"')
            completion = self.cache[str((system_instr, user_instr))]
            if not silent:
                print(f'Returning response: "{completion}"')
            return completion
        else:
            if not silent:
                print(f'Submitting prompt to LLaMA: "{user_instr}"')

            try:
                
                messages = [
                    {"role": "system", "content": system_instr},
                    {"role": "user", "content": user_instr},
                ]
                
                # Generate output using the pipeline
                outputs = self.pipeline(
                    messages,
                    max_new_tokens=512,  # Adjust as needed
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=self.pipeline.tokenizer.pad_token_id,
                    do_sample=True,
                )

                completion = outputs[0]["generated_text"][-1]["content"]

            except Exception as e:
                print(f"Error during prompt submission: {e}")
                return ""

            # Cache the result if caching is enabled
            if self.cache_path:
                self.cache[str((system_instr, user_instr))] = completion
                json.dump(self.cache, open(self.cache_path, "w"), indent=4)

            if not silent:
                print(f'Returning response: "{completion}"')
            return completion

    def get_principal_noun(self, caption):
        system_instr = (f'Suppose that you have an image caption describing a scene. What is the name of the most important '
                        f'object in this scene? Answer only with one word, the name of the object.\n"')
        user_instr = system_instr + (f'Answer only with one word, the name of the object. Caption: "{caption}."')
        response = self.submit_prompt(system_instr=system_instr, user_instr=user_instr)
        response = response.lower().replace(".", "")
        return response

    def get_movable_obj_idx(self, user_instr, obj_captions):
        system_instr = (f'You are a robot. There are some objects in the scene. The user gives you an instruction. '
                        f'Decide which one object the user wants the robot to move. Do not include any objects which should remain '
                        f'unmoved (e.g. containers). Below, a description is given for each of the objects, with a correspondind index. '
                        f'Do not include an explanation or commentary in your response. Your answer must be only one number, which corresponds to the index '
                        f'of the object which should be moved.\n')
        user_instr = system_instr + (f'User instruction: "{user_instr}"\n')
        assert obj_captions[0] == "__background__"
        for i, caption in enumerate(obj_captions[1:]):  # Skip background
            user_instr += f'Object {i + 1}: "{caption}"\n'

        response = self.submit_prompt(system_instr=system_instr, user_instr=user_instr)
        movable_idx = int(re.findall(r'\d+', response)[0])
        return movable_idx

    def get_relevant_obj_idxs(self, scene_caption, obj_captions, movable_obj_idx):
        system_instr = (f'Suppose you are a robot. You are given a caption of a scene. Below, you are also given some object '
                        f'descriptions. For each object description, determine whether it is a distractor object. Return a separate line '
                        f'for each object containing Yes or No, where Yes means that it is a distractor. A distractor object is one '
                        f'which is not one of the objects mentioned in the scene caption. Be careful that the object descriptions '
                        f'are based on low-quality images where the text is not easily identified, so ignore that part of the object descriptions. '
                        f'If the object description could plausibly describe an object in the caption, you must return No. Each line in the response '
                        f'should have the format: Object <number>: Yes if the object is a distractor; Object <number>: No if the object is not a distractor. '
                        f'Only if none of the objects in the scene are distractors, the final line should just be one word: "None".\n')
        assert obj_captions[0] == "__background__"

        # Temporarily swap object at idx 1 with movable object, so that LLM sees movable first.
        obj_captions = obj_captions.copy()
        obj_captions[1], obj_captions[movable_obj_idx] = obj_captions[movable_obj_idx], obj_captions[1]

        user_instr = system_instr + (f'Scene caption: "{scene_caption}"\n')
        for i, caption in enumerate(obj_captions[1:]):  # Skip background
            user_instr += f'Object {i + 1}: "{caption}"\n'

        response = self.submit_prompt(system_instr=system_instr, user_instr=user_instr)
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
            obj_captions), f"Error: LLM returned wrong number of decisions for distractor status for objects. Expected {len(obj_captions)}, got {len(decisions)+1}."
        return relevant_idxs

    def aggregate_captions_for_obj(self, captions, silent=True):
        system_instr = (f'Suppose we have captured many images of an object across different views. For each view, we have asked a network to '
            f'caption the image. Some captions may be wrong, and there may be some other objects in view accidentally (e.g. inside, nearby or '
            f'on top of the main object) which you must ignore. Please aggregate the caption information from across views, and write '
            f'a caption which best describes the main object being captured. If the object can be a couple of things, mention them both, '
            f'using a comma to separate multiple options. Do not include quotation marks and/or break lines in your response.'
            f'The caption should be concise and to the point, and you should not provide any explanation or commentary.\n')
        user_instr = system_instr + (f'List of captions:\n')
        for caption in captions:
            user_instr += f'"{caption}"\n'

        response = self.submit_prompt(system_instr=system_instr, user_instr=user_instr, silent=silent)
        return response

    def parse_instr(self, user_instr):
        system_instr = (f'Suppose you are a robot. You are given an instruction from a user. First, you need to extract the goal caption from the prompt. '
            f'This is a description of the desired state after the user instruction has been executed. E.g. if the instruction is "shove the X under Y", '
            f'the goal caption would be "an X under a Y". Also, you should extract a normalising caption from the goal caption. This will list the objects '
            f'mentioned in the goal caption but without any spatial relations. Your first returned line should be the goal caption (the line should begin with '
            f'"Goal caption: "), and the second line should be the normalising caption (the line should begin with "Normalising caption: "). No quotation marks '
            f'needed. E.g. if the goal caption is "an X under a Y", then the normalising caption would be "an X and a Y". If the goal caption is "big Xs in the style '
            f'of something", then the normalising caption is just "big Xs". However, you should keep spatial relations if they refer to a table, because objects will always '
            f'be above table level. E.g. if the goal caption is "Xs arranged in a grid on a plastic table", then the normalising caption would be "Xs on a plastic table".\n')
        user_instr = system_instr + (f'User instruction: "{user_instr}"\n')
        response = self.submit_prompt(system_instr=system_instr, user_instr=user_instr)
        goal_caption, norm_caption = response.split("\n")
        goal_caption = goal_caption.replace("Goal caption: ", "")
        norm_caption = norm_caption.replace("Normalising caption: ", "")
        return goal_caption, norm_caption


if __name__ == '__main__':
    # Test the langmodel
    model = LangModel(read_cache=False)
    import time
    start = time.time()
    result = model.submit_prompt("You are an helpful assistant.", "How many R are in strawberry?")
    end = time.time()
    print(f'Time taken: {int(end - start)} seconds')
