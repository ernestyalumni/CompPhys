class Stack:
    """ @class Stack  
        @details  Stack follows LIFO. Last in (append), First Out (pop), like stack of plates
    """
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self,item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        L = len(self.items)
        return self.items[L-1] # 0-based Python counting

    def size(self):
        L = len(self.items)
        return L

class Queue:
    """ @class Queue  
        @details  Stack follows FIFO. First in (insert), First Out (pop); elements removed in same order they came in 
    """
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self,item):
        self.items.append(item)

    def pop(self):
        return self.remove( self.items[0])

    def peek(self):
        L = len(self.items)
        return self.items[L-1] # 0-based Python counting

    def size(self):
        L = len(self.items)
        return L


