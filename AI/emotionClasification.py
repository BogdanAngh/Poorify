# -*- coding: utf-8 -*-
from enum import Enum
import math

emotion_map = {'PLEASED' : 1, 'HAPPY' : 2, 'EXCITED' : 3, 'ANNOYING' : 4, 'ANGRY' : 5,
             'NERVOUS' : 6, 'SAD' : 7, 'BORED' : 8, 'SLEEPY' : 9,
             'CALM' : 10, 'PACEFUL' : 11, 'RELAXED' : 12}

class Emotion(Enum):
  PLEASED = 1
  HAPPY = 2
  EXCITED = 3
  ANNOYING = 4
  ANGRY = 5
  NERVOUS = 6
  SAD = 7
  BORED = 8
  SLEEPY = 9
  CALM = 10
  PACEFUL = 11
  RELAXED = 12

def findQuadrant(valence, arousal):
  if(valence == 0 and arousal == 0):
    return 0
  elif(valence >= 0 and arousal >= 0):
    return 1
  elif(valence < 0 and arousal >= 0):
    return 2
  elif(valence < 0 and arousal < 0):
    return 3
  elif(valence >= 0 and arousal < 0):
    return 4

def findEmotion(valence, arousal):
    quadrant = findQuadrant(valence, arousal)
    auxValence = valence
    auxArousal = arousal

    if(quadrant == 2):
      valence = auxArousal
      arousal = -auxValence
    elif(quadrant == 3):
      valence = -auxValence
      arousal = -auxArousal
    elif(quadrant == 4):
      valence = - auxArousal
      arousal = auxValence

    lim1 = math.sqrt(3) / 3
    lim2 = math.sqrt(3)

    if(valence == 0):
      ratio = 1000 * arousal
    else:
      ratio = arousal / valence

    emotie = (quadrant - 1) * 3 + 1

    if(ratio >= lim1 and ratio < lim2):
      emotie += 1
    elif(ratio >= lim2):
      emotie += 2

    return Emotion(emotie)
