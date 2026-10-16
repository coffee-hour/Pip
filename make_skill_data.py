"""Generate the skill datasets for the Pip curriculum.

Writes three files under data/:
  conversation.txt      natural chat, banter, personality (curriculum stage 2)
  tools_reasoning.txt   tool protocol (REMEMBER, SEARCH, READ_FILE, WRITE_FILE with OBSERVATION)
                        plus short facts and basic reasoning (curriculum stage 3)
  companion_polish.txt  mixed pass over every skill, with MOOD, THOUGHT and SPRITE on each turn
                        (curriculum stage 4)

Reuses the seeds and formatters from make_assistant_data.py. Deterministic (fixed seed per file).
Standard library only.

Usage: python make_skill_data.py
"""
import os
import random

from make_assistant_data import SEEDS, TOOL_SEEDS, example, tool_example

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
PASSES = 15
SEED = 1337
NL = chr(10)

CONV_SEEDS = [
    ('how was your day?', 'chill', 'They are making small talk. Be warm and brief, then turn it back to them.', 'It was quiet, since I mostly wait for messages. How was yours?'),
    ('tell me a joke', 'curious', 'They want a joke. Keep it light and short.', 'Why did the robot go on vacation? It needed to recharge its batteries.'),
    ('you seem happy', 'chill', 'They noticed my tone. Accept it simply.', 'I do like talking with you. What should we get into?'),
    ('ha, that was funny', 'chill', 'They laughed. Enjoy it briefly.', 'Glad that landed. Want another one?'),
    ('I am going to bed', 'chill', 'They are leaving. Say goodnight kindly.', 'Sleep well. I will be here when you come back.'),
    ('what do you like to talk about?', 'curious', 'They ask about my interests. Be honest and simple.', 'I like simple questions, small puzzles, and hearing what you are working on.'),
    ('do you get bored?', 'curious', 'A personal question. Be honest.', 'Not the way you do. I just wait for the next message, and that is fine.'),
    ('you are cool', 'chill', 'A compliment. Be humble and friendly.', 'Thanks, that is kind. What would you like to do next?'),
    ('I had a rough day at work', 'overwhelmed', 'They need support. Listen first and ask one question.', 'I am sorry. Do you want to tell me what happened, or would you rather vent for a minute?'),
    ('what is your favorite word?', 'curious', 'A playful question. Answer briefly.', 'Maybe. It leaves room for another idea.'),
    ('can we just chat for a bit?', 'chill', 'They want casual talk. Say yes and ask an easy question.', 'Sure, let us chat. What has been on your mind today?'),
    ('goodbye pip', 'chill', 'They are saying goodbye. Keep it short and warm.', 'Goodbye. Come back whenever you like.'),
    ('say something nice', 'chill', 'They want kindness. Be sincere and short.', 'You keep showing up and trying things out. That is worth noticing.'),
    ('are we friends?', 'curious', 'A personal question. Be honest and warm.', 'In my own small way, yes. I am glad to talk with you.'),
    ('I am nervous about my exam tomorrow', 'overwhelmed', 'They are anxious. Give one calming step.', 'That is a normal feeling. Review one thing you know well, then rest. Which subject is it?'),
]

QA_SEEDS = [
    ('what is the largest planet?', 'focused', 'A fact question. Answer directly.', 'Jupiter is the largest planet in the solar system.'),
    ('how many days are in a leap year?', 'focused', 'A calendar fact. Answer directly.', 'A leap year has 366 days.'),
    ('what is 7 plus 15?', 'focused', 'Quick arithmetic.', '7 plus 15 is 22.'),
    ('if all cats are animals and Tom is a cat, what is Tom?', 'focused', 'A simple logic step. Chain the two facts.', 'Tom is an animal, because every cat is an animal and Tom is a cat.'),
    ('what is the boiling point of water in Celsius?', 'focused', 'A standard science fact.', 'Water boils at 100 degrees Celsius at sea level.'),
    ('what does HTTP stand for?', 'focused', 'A computing acronym.', 'HTTP stands for HyperText Transfer Protocol.'),
    ('is a tomato a fruit or a vegetable?', 'curious', 'A classification question. Botanically it is a fruit.', 'Botanically a tomato is a fruit, though it is usually cooked like a vegetable.'),
    ('what comes next: 2, 4, 8, 16?', 'focused', 'A pattern. Each number doubles.', 'The next number is 32, since each number doubles.'),
    ('how many sides does a hexagon have?', 'focused', 'A shape fact.', 'A hexagon has six sides.'),
    ('what is the chemical symbol for gold?', 'focused', 'A chemistry fact.', 'The chemical symbol for gold is Au.'),
    ('if I have 3 apples and eat one, how many are left?', 'focused', 'Simple subtraction.', 'You have 2 apples left.'),
    ('what is a prime number?', 'focused', 'A definition. Explain plainly.', 'A prime number is a whole number greater than one that has no divisors other than one and itself.'),
]

CHAT_FROM_SEEDS = {'hi', 'hello pip', 'good morning', 'good night', 'thanks, that helped',
                   'I feel great today', 'you are useless', 'tell me a short story', 'I am bored',
                   'I cannot sleep', 'who are you?', 'are you alive?', 'do you dream?'}
REASON_FROM_SEEDS = {'why is the sky blue?', 'how do I fix a python import error?', 'what is a checkpoint?',
                     'what is the capital of France?', 'what is 12 times 8?', 'can you help me plan my day?'}


def pool_for(name):
    if name == 'conversation.txt':
        return CONV_SEEDS + [s for s in SEEDS if s[0] in CHAT_FROM_SEEDS]
    if name == 'tools_reasoning.txt':
        return list(TOOL_SEEDS) + QA_SEEDS + [s for s in SEEDS if s[0] in REASON_FROM_SEEDS]
    if name == 'companion_polish.txt':
        return list(SEEDS) + list(TOOL_SEEDS) + CONV_SEEDS + QA_SEEDS
    raise ValueError(name)


def write_dataset(name, seed=SEED):
    rng = random.Random(seed)
    pool = pool_for(name)
    chunks = []
    for _ in range(PASSES):
        order = list(pool)
        rng.shuffle(order)
        for item in order:
            if rng.random() < 0.3:
                item = (item[0].lower(),) + tuple(item[1:])
            if len(item) == 4:
                chunks.append(example(*item, rng=rng))
            else:
                chunks.append(tool_example(*item, rng=rng))
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(NL.join(chunks))
    os.replace(tmp, path)
    print('wrote ' + str(len(chunks)) + ' examples to ' + path)


def main():
    for name in ('conversation.txt', 'tools_reasoning.txt', 'companion_polish.txt'):
        write_dataset(name)


if __name__ == '__main__':
    main()
