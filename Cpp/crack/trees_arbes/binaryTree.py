""" @name binaryTree.py
"""

class binaryTree(object):
    """ @class binaryTree
        @note data structure of this particular class is really a node
        a node has a left child and right child (at most)
    """
    def __init__(self,key=None):
        self.val = key
        self.l = None
        self.r = None

    def isLeaf(self,v): # v for vertex or node
        return (v.l is None and v.r is None)

    def insertl(self,newkey):
        if self.l == None:
            self.l = binaryTree(newkey)
        else:
            l = binaryTree(newkey)
            l.l = self.l # push down left child to the left
            self.l = l
    
    def insertr(self,newkey):
        if self.r == None:
            self.r = binaryTree(newkey)
        else:
            r = binaryTree(newkey)
            r.r = self.r # push down right child to the right
            self.r = r 
    
    def inorder(self):
        if self.l:
            self.l.inorder()
        print(self.val)
        if self.r:
            self.r.inorder()

    def postorder(self):
        if self.l:
            self.l.postorder()
        if self.r:
            self.r.postorder()
        print(self.val)

    def preorder(self):
        print(self.key)
        if self.l:
            self.l.preorder()
        if self.r:
            self.r.preorder()

            

