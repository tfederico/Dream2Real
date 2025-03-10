import os
import json
import re
import ollama

class LangModel():
    def __init__(self, model_name="llama3.1", read_cache=True, cache_path=""):
        self.check_cache = read_cache
        self.cache_path = cache_path
        self.model_name = model_name
        
        # No need for explicit initialization with ollama library

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
                print(f'Submitting prompt to Ollama: "{user_instr}"')

            try:
                # Format messages for Ollama
                messages = [
                    {"role": "system", "content": system_instr},
                    {"role": "user", "content": user_instr}
                ]
                
                # Generate output using ollama client
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": 512
                    }
                )
                
                # Extract completion from response
                completion = response["message"]["content"]

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
        user_instr = (f'Answer only with one word, the name of the object. Caption: "{caption}."')
        response = self.submit_prompt(system_instr=system_instr, user_instr=user_instr)
        response = response.lower().replace(".", "")
        return response

    def get_movable_obj_idx(self, user_instr, obj_captions):
        system_instr = (f'You are a robot. There are some objects in the scene. The user gives you an instruction. '
                        f'Decide which one object the user wants the robot to move. Do not include any objects which should remain '
                        f'unmoved (e.g. containers). Below, a description is given for each of the objects, with a correspondind index. '
                        f'Do not include an explanation or commentary in your response. Your answer must be only one number, which corresponds to the index '
                        f'of the object which should be moved.\n')
        user_instr = (f'User instruction: "{user_instr}"\n')
        assert obj_captions[0] == "__background__"
        for i, caption in enumerate(obj_captions[1:]):  # Skip background
            user_instr += f'Object {i + 1}: "{caption}"\n'

        response = self.submit_prompt(system_instr=system_instr, user_instr=user_instr)
        movable_idx = int(re.findall(r'\d+', response)[0])
        return movable_idx

    def get_relevant_obj_idxs(self, scene_caption, obj_captions, movable_obj_idx):
        system_instr = (f'Suppose you are a robot. You are given a caption of a scene and an object description. '
                        f'Determine whether this object is relevant to the scene caption. An object is relevant if it is '
                        f'mentioned in the scene caption. Be quite strict about this. Be careful that the object descriptions are based on '
                        f'low-quality images where text may not be easily identified, so ignore that part of the object '
                        f'descriptions. If the object description could plausibly describe an object in the caption, '
                        f'you must return Yes. Do not include any explanation or commentary in your response.\n')
        assert obj_captions[0] == "__background__"

        # Temporarily swap object at idx 1 with movable object, so that LLM sees movable first.
        obj_captions = obj_captions.copy()
        obj_captions[1], obj_captions[movable_obj_idx] = obj_captions[movable_obj_idx], obj_captions[1]

        relevant_idxs = [movable_obj_idx]  # Movable always relevant
        
        # Check each object individually
        for i, caption in enumerate(obj_captions[1:], start=1):  # Skip background
            if i == 1:  # Skip movable object since it's always relevant
                continue
                
            user_instr = (f'Scene caption: "{scene_caption}"\n')
            user_instr += f'Object description: "{caption}"\n'

            response = self.submit_prompt(system_instr=system_instr, user_instr=user_instr)
            
            if 'Yes' in response:
                relevant_idx = 1 if i == movable_obj_idx else i
                relevant_idxs.append(relevant_idx)

        return relevant_idxs

    def aggregate_captions_for_obj(self, captions, silent=True):
        system_instr = (f'Suppose we have captured many images of an object across different views. For each view, we have asked a network to '
            f'caption the image. Some captions may be wrong, and there may be some other objects in view accidentally (e.g. inside, nearby or '
            f'on top of the main object) which you must ignore. Please aggregate the caption information from across views, and write '
            f'a caption which best describes the main object being captured. If the object can be a couple of things, mention them both, '
            f'using a comma to separate multiple options. Do not include quotation marks and/or break lines in your response.'
            f'The caption should be concise and to the point, and you should not provide any explanation or commentary.\n')
        user_instr = (f'List of captions:\n')
        for caption in captions:
            user_instr += f'"{caption}"\n'

        response = self.submit_prompt(system_instr=system_instr, user_instr=user_instr, silent=silent)
        # limit response to 2 options, LLM is chatty
        response = response.split(",")[:2]
        response = ", ".join(response)
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
        user_instr = (f'User instruction: "{user_instr}"\n')
        response = self.submit_prompt(system_instr=system_instr, user_instr=user_instr)
        goal_caption, norm_caption = response.split("\n")
        goal_caption = goal_caption.replace("Goal caption: ", "")
        norm_caption = norm_caption.replace("Normalising caption: ", "")
        return goal_caption, norm_caption
    
    def stop_ollama(self):
        os.system(f"ollama stop {self.model_name}")


if __name__ == '__main__':
    # Test the langmodel
    model = LangModel(read_cache=False)
    import time
    start = time.time()
    result = model.submit_prompt("You are an helpful assistant.", "How many R are in strawberry?")
    end = time.time()
    print(f'Time taken: {int(end - start)} seconds')
