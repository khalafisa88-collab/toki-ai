import numpy as np
import random

class MatrixCore:
    def __init__(self, vector_size=7):
        self.vector_size = vector_size
        self.word_vectors = {}
        self.memory = []
        self.transform_matrix = np.eye(vector_size)

    def add_word(self, word, vector):
        if len(vector) != self.vector_size:
            raise ValueError(f"بردار باید {self.vector_size} بعد داشته باشد")
        self.word_vectors[word] = np.array(vector)

    def sentence_to_vector(self, sentence):
        words = sentence.lower().split()
        if not words:
            return np.zeros(self.vector_size)
        vectors = [self.word_vectors.get(w, np.zeros(self.vector_size)) for w in words]
        return np.mean(vectors, axis=0)

    def set_transform_matrix(self, matrix):
        if matrix.shape != (self.vector_size, self.vector_size):
            raise ValueError("ابعاد ماتریس تبدیل نامعتبر است")
        self.transform_matrix = matrix

    def generate_response(self, user_input):
        self.memory.append(user_input)
        if len(self.memory) > 10:
            self.memory.pop(0)
        input_vector = self.sentence_to_vector(user_input)
        if np.all(input_vector == 0):
            output_vector = np.zeros(self.vector_size)
        else:
            output_vector = np.dot(self.transform_matrix, input_vector)
        self.memory.append(output_vector.tolist())
        return output_vector

    def clear_memory(self):
        self.memory = []

    def get_memory(self):
        return self.memory

def add_all_words(core):
    all_words = {
        "akesi": [1, 0, 1, 0, 0, 0, 0],
        "ala": [0, 1, 0, 0, 1, 0, 0],
        "alasa": [1, 0, 0, 1, 0, 0, 0],
        "ale": [0, 1, 1, 0, 1, 0, 0],
        "anpa": [1, 1, 0, 1, 0, 0, 0],
        "ante": [0, 0, 1, 1, 0, 0, 0],
        "anu": [1, 0, 0, 0, 1, 0, 0],
        "awen": [0, 1, 0, 0, 0, 0, 0],
        "e": [1, 1, 0, 1, 1, 0, 0],
        "en": [0, 0, 1, 0, 0, 0, 0],
        "esun": [1, 0, 1, 0, 1, 0, 0],
        "ijo": [0, 1, 0, 1, 0, 0, 0],
        "ike": [1, 1, 1, 0, 0, 0, 0],
        "ilo": [0, 0, 0, 1, 0, 0, 0],
        "insa": [1, 0, 1, 1, 1, 0, 0],
        "jaki": [0, 1, 0, 0, 0, 0, 0],
        "jan": [1, 1, 0, 1, 0, 0, 0],
        "jelo": [0, 0, 1, 0, 1, 0, 0],
        "jo": [1, 0, 0, 1, 0, 0, 0],
        "kala": [0, 1, 1, 1, 0, 0, 0],
        "kalama": [1, 0, 0, 0, 1, 0, 0],
        "kama": [0, 1, 1, 0, 0, 0, 0],
        "kasi": [1, 1, 0, 0, 1, 0, 0],
        "ken": [0, 0, 1, 0, 0, 0, 0],
        "kepeken": [1, 0, 1, 0, 1, 0, 0],
        "kili": [0, 1, 0, 1, 0, 0, 0],
        "kin": [1, 1, 1, 1, 0, 0, 0],
        "kipisi": [0, 0, 0, 1, 0, 0, 0],
        "kiwen": [1, 0, 1, 1, 1, 0, 0],
        "ko": [0, 1, 0, 0, 0, 0, 0],
        "kon": [1, 1, 0, 1, 0, 0, 0],
        "kule": [0, 0, 1, 0, 1, 0, 0],
        "kulupu": [1, 0, 0, 1, 0, 0, 0],
        "kute": [0, 1, 1, 1, 0, 0, 0],
        "la": [1, 0, 0, 0, 1, 0, 0],
        "lape": [0, 1, 0, 0, 0, 0, 0],
        "laso": [1, 1, 0, 1, 1, 0, 0],
        "lawa": [0, 0, 1, 0, 0, 0, 0],
        "len": [1, 0, 1, 0, 1, 0, 0],
        "lete": [0, 1, 0, 1, 0, 0, 0],
        "li": [1, 1, 1, 1, 0, 0, 0],
        "lili": [0, 0, 0, 1, 0, 0, 0],
        "linja": [1, 0, 1, 1, 1, 0, 0],
        "lipu": [0, 1, 0, 0, 0, 0, 0],
        "loje": [1, 1, 0, 1, 0, 0, 0],
        "lon": [0, 0, 1, 0, 1, 0, 0],
        "luka": [1, 0, 0, 1, 0, 0, 0],
        "lukin": [0, 1, 1, 1, 0, 0, 0],
        "lupa": [1, 0, 0, 0, 1, 0, 0],
        "ma": [0, 1, 0, 0, 0, 0, 0],
        "mama": [1, 1, 0, 1, 1, 0, 0],
        "mani": [0, 0, 1, 0, 0, 0, 0],
        "meli": [1, 0, 1, 0, 1, 0, 0],
        "mi": [0, 1, 0, 1, 0, 0, 0],
        "mije": [1, 1, 1, 1, 0, 0, 0],
        "moku": [0, 0, 0, 1, 0, 0, 0],
        "moli": [1, 0, 1, 1, 1, 0, 0],
        "monsi": [0, 1, 0, 0, 0, 0, 0],
        "mu": [1, 1, 0, 1, 0, 0, 0],
        "mun": [0, 0, 1, 0, 1, 0, 0],
        "musi": [1, 0, 0, 1, 0, 0, 0],
        "mute": [0, 1, 1, 1, 0, 0, 0],
        "namako": [1, 0, 0, 0, 1, 0, 0],
        "nanpa": [0, 1, 0, 0, 0, 0, 0],
        "nasa": [1, 1, 0, 1, 1, 0, 0],
        "nasin": [0, 0, 1, 0, 0, 0, 0],
        "nena": [1, 0, 1, 0, 1, 0, 0],
        "ni": [0, 1, 0, 1, 0, 0, 0],
        "nimi": [1, 1, 1, 1, 0, 0, 0],
        "noka": [0, 0, 0, 1, 0, 0, 0],
        "o": [1, 0, 1, 1, 1, 0, 0],
        "olin": [0, 1, 0, 0, 0, 0, 0],
        "ona": [1, 1, 0, 1, 0, 0, 0],
        "open": [0, 0, 1, 0, 1, 0, 0],
        "pakala": [1, 0, 0, 1, 0, 0, 0],
        "pali": [0, 1, 1, 1, 0, 0, 0],
        "palisa": [1, 0, 0, 0, 1, 0, 0],
        "pana": [0, 1, 0, 0, 0, 0, 0],
        "pi": [1, 1, 0, 1, 1, 0, 0],
        "pilin": [0, 0, 1, 0, 0, 0, 0],
        "pimeja": [1, 0, 1, 0, 1, 0, 0],
        "pini": [0, 1, 0, 1, 0, 0, 0],
        "pipi": [1, 1, 1, 1, 0, 0, 0],
        "poka": [0, 0, 0, 1, 0, 0, 0],
        "pona": [1, 0, 1, 1, 1, 0, 0],
        "sama": [0, 1, 0, 0, 0, 0, 0],
        "seli": [1, 1, 0, 1, 0, 0, 0],
        "selo": [0, 0, 1, 0, 1, 0, 0],
        "seme": [1, 0, 0, 1, 0, 0, 0],
        "sewi": [0, 1, 1, 1, 0, 0, 0],
        "sijelo": [1, 0, 0, 0, 1, 0, 0],
        "sike": [0, 1, 0, 0, 0, 0, 0],
        "sin": [1, 1, 0, 1, 1, 0, 0],
        "sina": [0, 0, 1, 0, 0, 0, 0],
        "sinpin": [1, 0, 1, 0, 1, 0, 0],
        "sitelen": [0, 1, 0, 1, 0, 0, 0],
        "sona": [1, 1, 1, 1, 0, 0, 0],
        "soweli": [0, 0, 0, 1, 0, 0, 0],
        "suli": [1, 0, 1, 1, 1, 0, 0],
        "suno": [0, 1, 0, 0, 0, 0, 0],
        "supa": [1, 1, 0, 1, 0, 0, 0],
        "suwi": [0, 0, 1, 0, 1, 0, 0],
        "tan": [1, 0, 0, 1, 0, 0, 0],
        "taso": [0, 1, 1, 1, 0, 0, 0],
        "tawa": [1, 0, 0, 0, 1, 0, 0],
        "telo": [0, 1, 0, 0, 0, 0, 0],
        "tenpo": [1, 1, 0, 1, 1, 0, 0],
        "toki": [0, 0, 1, 0, 0, 0, 0],
        "tu": [1, 0, 1, 0, 1, 0, 0],
        "unpa": [0, 1, 0, 1, 0, 0, 0],
        "uta": [1, 1, 1, 1, 0, 0, 0],
        "utala": [0, 0, 0, 1, 0, 0, 0],
        "walo": [1, 0, 1, 1, 1, 0, 0],
        "wan": [0, 1, 0, 0, 0, 0, 0],
        "waso": [1, 1, 0, 1, 0, 0, 0],
        "wawa": [0, 0, 1, 0, 1, 0, 0],
        "weka": [1, 0, 0, 1, 0, 0, 0],
        "wile": [0, 1, 1, 1, 0, 0, 0]
    }
    for word, vec in all_words.items():
        core.add_word(word, vec)

def run_ui(vector_size=7):
    core = MatrixCore(vector_size)
    add_all_words(core)
    print("🤖 Matrix ChatBot (v1.0)")
    print("دستورات: /exit, /clear, /show")
    print("-" * 40)

    while True:
        user_input = input("شما: ").strip()
        if user_input.lower() in ["/exit", "exit"]:
            print("خداحافظ!")
            break
        elif user_input.lower() == "/clear":
            core.clear_memory()
            print("حافظه پاک شد!")
            continue
        elif user_input.lower() == "/show":
            print("--- تاریخچه ---")
            for item in core.get_memory():
                print(item)
            print("-" * 40)
            continue
        elif user_input == "":
            continue

        output_vector = core.generate_response(user_input)
        print(f"بردار خروجی: {output_vector}")
        print("-" * 40)

if __name__ == "__main__":
    run_ui(vector_size=7)
