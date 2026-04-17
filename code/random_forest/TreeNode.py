class TreeNode:

    def __init__(self, attribute: str, threshold: int, level: int):
        self.attribute = attribute
        self.threshold = threshold
        self.level = level

        # These initialization values are for type consistency and are never used
        self.under = DecideMedium(1)
        self.over = DecideMedium(1)

    def decide(self, sample):
        if sample[self.attribute] <= self.threshold:
            return self.under.decide(sample)
        else:
            return self.over.decide(sample)

    def as_code(self):
        tabs = "".join('\t' for _ in range(self.level))
        out = f"{tabs}if sample.{self.attribute} <= {self.threshold}:\n"
        out += self.under.as_code()
        out += f"{tabs}else:\n"
        out += self.over.as_code()
        return out
        
class DecideLow:

    def __init__(self, level):
        self.level = level
    
    def decide(self, sample):
        return 0

    def as_code(self):
        tabs = "".join('\t' for _ in range(self.level))
        return f"{tabs}return 0\n"
    
class DecideMedium:

    def __init__(self, level):
        self.level = level

    def decide(self, sample):
        return 1

    def as_code(self):
        tabs = "".join('\t' for _ in range(self.level))
        return f"{tabs}return 1\n"
    
class DecideHigh:

    def __init__(self, level):
        self.level = level

    def decide(self, sample):
        return 2

    def as_code(self):
        tabs = "".join('\t' for _ in range(self.level))
        return f"{tabs}return 2\n"