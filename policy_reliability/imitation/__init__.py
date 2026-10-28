"""
Imitation learning from scratch on PushT (state observations), trained on
LeRobot's human demonstrations and evaluated with the same harness.

Variants compared on identical evaluation scenes:
  * single-step behaviour cloning (predict the next action)
  * action chunking (predict the next k actions, execute them open-loop)
  * action chunking + temporal ensembling (re-plan every step, average the
    overlapping predictions for the current step), as in ACT
"""
