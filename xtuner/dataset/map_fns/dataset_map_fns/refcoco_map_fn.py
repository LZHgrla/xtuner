from xtuner.utils import DEFAULT_IMAGE_TOKEN


def refcoco_map_fn(data):
    conversation = {}
    input: str = data['instruction_input']
    input = input.replace('<Img><ImageHere></Img> ',
                          DEFAULT_IMAGE_TOKEN+'\n').strip()
    conversation['input'] = input
    conversation['output'] = data['answer']
    conversation['system'] = ""
    return {'conversation': [conversation]}
