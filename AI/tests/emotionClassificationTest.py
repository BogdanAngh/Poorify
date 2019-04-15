import emotionClasification as ec

assert ec.findEmotion(1,0.1) == ec.Emotion.PLEASED, "Emotion PLEASED was not clasified correctly"
assert ec.findEmotion(1,1) == ec.Emotion.HAPPY, "Emotion HAPPY was not clasified correctly"
assert ec.findEmotion(1,2) == ec.Emotion.EXCITED, "Emotion EXCITED was not clasified correctly"
assert ec.findEmotion(-0.1,1) == ec.Emotion.ANNOYING, "Emotion ANNOYING was not clasified correctly"
assert ec.findEmotion(-1,1) == ec.Emotion.ANGRY, "Emotion ANGRY was not clasified correctly"
assert ec.findEmotion(-2,1) == ec.Emotion.NERVOUS, "Emotion NERVOUS was not clasified correctly"
assert ec.findEmotion(-1,-0.1) == ec.Emotion.SAD, "Emotion SAD was not clasified correctly"
assert ec.findEmotion(-1,-1) == ec.Emotion.BORED, "Emotion BORED was not clasified correctly"
assert ec.findEmotion(-1,-2) == ec.Emotion.SLEEPY, "Emotion SLEEPY was not clasified correctly"
assert ec.findEmotion(0.1,-1) == ec.Emotion.CALM, "Emotion CALM was not clasified correctly"
assert ec.findEmotion(1,-1) == ec.Emotion.PACEFUL, "Emotion PACEFUL was not clasified correctly"
assert ec.findEmotion(2,-1) == ec.Emotion.RELAXED, "Emotion RELAXED was not clasified correctly"


print("Tests for emotionClasification have passed")
