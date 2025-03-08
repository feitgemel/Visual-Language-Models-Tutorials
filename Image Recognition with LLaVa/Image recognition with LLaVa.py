import ollama

imagePath1 = "Visual-Language-Models-Tutorials/Image Recognition with LLaVa/Parrot.jpg"
imagePath2 = "Visual-Language-Models-Tutorials/Image Recognition with LLaVa/Rahaf.jpg"

result = ollama.chat(
    model = "llava:13b",
    messages=[
        {'role' : 'user',
         'content' : "Describe this image",
         'images' : [imagePath1]
         }
    ]
)

#print(result)
print("Describe this image : ")
resultText = result['message']['content']
print(resultText)
print("**********************************************************************")

# let's try another image and different prompt:

result = ollama.chat(
    model = "llava:13b",
    messages=[
        {'role' : 'user',
         'content' : "What is the color of the shirt of the woman in the center ? ",
         'images' : [imagePath2]
         }
    ]
)

print("What is the color of the shirt of the woman in the center ? ")
resultText = result['message']['content']
print(resultText)
print("**********************************************************************")


# let's try another image and different prompt:

result = ollama.chat(
    model = "llava:13b",
    messages=[
        {'role' : 'user',
         'content' : "Generate 5 keywords describing this image",
         'images' : [imagePath2]
         }
    ]
)

print("Generate 5 keywords describing this image : ")
resultText = result['message']['content']
print(resultText)
print("**********************************************************************")


# let's try another image and different prompt:

result = ollama.chat(
    model = "llava:13b",
    messages=[
        {'role' : 'user',
         'content' : "What is written on the sign behind the man ?",
         'images' : [imagePath1]
         }
    ]
)

print("What is written on the sign behind the man ? ")
resultText = result['message']['content']
print(resultText)
print("**********************************************************************")



