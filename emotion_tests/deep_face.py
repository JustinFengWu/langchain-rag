from deepface import DeepFace

objs = DeepFace.analyze(
  img_path = "./pictures/happy-guy3.jpg", 
  actions = ['emotion'],
)

print(objs)