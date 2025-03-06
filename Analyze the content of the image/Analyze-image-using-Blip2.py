from transformers import Blip2ForConditionalGeneration, Blip2Processor
import torch

from PIL import Image
import requests

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# https://huggingface.co/Salesforce/blip2-opt-2.7b

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

model.to(device)

url = "https://images.pexels.com/photos/12426042/pexels-photo-12426042.jpeg"
image = Image.open(requests.get(url, stream=True).raw)



# Step1 - What the model sees :

inputs = processor(images=image , return_tensors='pt' , text="")
inputs.to(device)
generate_ids = model.generate(**inputs, max_new_tokens=50 )
genrated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()

print("**********************************************")
print("What the model sees :" + genrated_text)
print("**********************************************")


# step 2 - Ask a question : what is the color of the couch ?
prompt = "Question: What is the color of the couch ? Answer:"
inputs = processor(images=image , return_tensors='pt' , text=prompt)
inputs.to(device)
generate_ids = model.generate(**inputs, max_new_tokens=50 )
genrated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()

print("**********************************************")
print("What is the color of the couch ? :" + genrated_text)
print("**********************************************")

# Another question :

prompt = "Question: How many cats ? Answer: "
inputs = processor(images=image , return_tensors='pt' , text=prompt)
inputs.to(device)
generate_ids = model.generate(**inputs, max_new_tokens=50 )
genrated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()

print("**********************************************")
print("How many cats ? :" + genrated_text)
print("**********************************************")



image.show()