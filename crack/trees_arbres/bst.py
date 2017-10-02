"""
	@file   : bst.py
	@brief  : Binary Search Tree (bst)
    @details : BinarySearchTree TreeNode
	@author : Ernest Yeung	ernestyalumni@gmail.com
	@date   : 20170918
	@ref    : cf. http://interactivepython.org/runestone/static/pythonds/Trees/SearchTreeImplementation.html 
                     https://github.com/bnmnetp/pythonds/blob/master/trees/bst.py
  
	If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
	
	https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
	
	which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
	Otherwise, I receive emails and messages on how all my (free) material on 
	physics, math, and engineering have helped students with their studies, 
	and I know what it's like to not have money as a student, but love physics 
	(or math, sciences, etc.), so I am committed to keeping all my material 
	open-source and free, whether or not 
	sufficiently crowdfunded, under the open-source MIT license: 
	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
	Just don't be an asshole and not give credit where credit is due.  
	Peace out, never give up! -EY
"""
class BinarySearchTree:
    """
    @class      BinarySearchTree  
    @date       20170918
    @details    Implement a binary search tree with the following interface functions:  
                __contains__(y) <==> y in x
                __getitem__(y) <==> x[y]
                __init__()
                __len__() <==> len(x)
                __setitem__(k,v) <==> x[k] = v
                clear()
                get(k)
                items()
                keys()
                values()
                put(k,v)
                in
                del <==>
    """

    def __init__(self):
        """ @fn __init__
            @note to initialize an instance, e.g. x = MyClass()
        """
        self.root = None
        self.size = 0
    
    def put(self,key,val):
        """ @fn put
            @brief 
        """
        if self.root:   # check to see if tree already has a root
            self._put(key,val,self.root) # root node already in place,& so call private, recursive helper function _put
        else: # create a new TreeNode and install it as the root of the tree
            self.root = TreeNode(key,val)
        self.size= self.size + 1

    def _put(self,key,val,currentNode): 
        """ @fn _put
            @details - start at root of tree, search binary tree, comparing the new key to the key in current node
                        Compare key to insert (new) to key in current node; 
                          if key < currentNode.key, search left subtree
                     - when there's no left (or right) child to search, we've found position in tree where 
                        new node should be installed
        """
        if key < currentNode.key: # search left subtree case
            if currentNode.hasLeftChild():
                self._put(key,val,currentNode.leftChild)
            else: 
                currentNode.leftChild=TreeNode(key,val,parent=currentNode)
        else:
            if currentNode.hasRightChild():
                self._put(key,val,currentNode.rightChild)
            else:
                currentNode.rightChild=TreeNode(key,val,parent=currentNode)
    
    def __setitem__(self,k,v):
        """ @fn __setitem__
            @brief overload [] operator
            @note to set a value by its key, e.g. x[key] = value , and Python calls x.__setitem__(key,value)
        """
        self.put(k,v)

    def get(self,key):
        if self.root:
            res = self._get(key,self.root)
            if res:
                return res.payload
            else:
                return None
        else:
            return None

    def _get(self,key,currentNode):
        if not currentNode:
            return None
        elif currentNode.key == key:
            return currentNode
        elif key < currentNode.key: # search the left subtree to try to find key
            return self._get(key,currentNode.leftChild)
        else: # search the right subtree to try to find key
            return self._get(key,currentNode.rightChild)

    def __getitem__(self,key):
        """ @fn __getitem__
            @note to get a value by its key, e.g. x[key] 
        """
        res = self.get(key)
        if res:
            return res
        else:
            raise KeyError('Error, key not in tree')

    def __contains__(self,key):
        if self._get(key,self.root):
            return True
        else:
            return False

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def __iter__(self):
        """ @fn __iter__
            @note to iterate through a sequence, e.g. iter(seq), and Python calls seq.__iter__()
        """
        return self.root.__iter__()

    def delete(self,key):
        if self.size > 1:
            nodeToRemove = self._get(key,self.root)  # find node to delete by searching tree
            if nodeToRemove:
                self.remove(nodeToRemove)
                self.size = self.size- 1
            else:
                raise KeyError('Error, key not in tree')
        elif self.size == 1 and self.root.key == key:
            self.root = None
            self.size = self.size - 1 
        else:
            raise KeyError('Error, key not in tree')

    def __delitem__(self,key):
        """ @fn __delitem__
            @brief to delete a key-value pair, e.g. del x[key], and Python calls x.__delitem__(key)
        """
        self.delete(key)

    def remove(self,currentNode):
        """ @fn remove
            @details  currentNode is to be deleted/removed
        """
        if currentNode.isLeaf(): # leaf # node to be deleted has no children
            if currentNode == currentNode.parent.leftChild: # remove from curreNode's parent
                currentNode.parent.leftChild = None 
            else: # remove from curreNode's parent
                currentNode.parent.rightChild = None
        elif currentNode.hasBothChildren(): # interior 
            succ = currentNode.findSuccessor() # search tree for node that can be used to replace 1 scheduled for deletion
            succ.spliceOut() # remove the successor; it goes directly to node we want to splice out and makes right change
            currentNode.key = succ.key
            curretNode.payload = succ.payload
        else: # this node has 1 child # if node has only 1 child, then simply promote child to take the place of parent
            if currentNode.hasLeftChild():
                if currentNode.isLeftChild(): # only need to update parent reference of left child to point to parent
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.LeftChild # update left child reference of parent 
                elif currentNode.isRightChild(): # need to update parent reference of left child
                    currentNode.leftChild.parent = currentNode.parent # point to parent of current node
                    currentNode.parent.rightChild = currentNode.leftChild # update right child reference of parent 
                else: # if current node has no parent (not a child), it must be root
                    currentNode.replaceNodeData(currentNode.leftChild.key, # replace using replaceNodeData
                                        currentNode.leftChild.payload,
                                        currentNode.leftChild.leftChild,
                                        currentNode.leftChild.rightChild)
            else:
                if currentNode.isLeftChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.rightChild
                elif currentNode.isRightChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.rightChild
                else:
                    currentNode.replaceNodeData(currentNode.rightChild.key, 
                                            currentNode.rightChild.payload,
                                            currentNode.rightChild.leftChild,
                                            currentNode.rightChild.rightChild)



class TreeNode:
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent
        self.balanceFactor = 0

    def hasLeftChild(self):
        return self.leftChild
    
    def hasRightChild(self):
        return self.rightChild 

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self
    
    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild and self.leftChild

    def replaceNodeData(self,key,value,lc,rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self
        
    def findSuccessor(self):
        """ @fn findSuccessor
            @brief 
            @details  Use binary search tree property that cause an inorder traversal to print out nodes in tree
                        from smallest to largest.  3 cases to consider 
                        1. if node has right child, then successor is smallest key in right subtree
                        2. if node has no right child and is left child of its parent, then parent is successor
                        3. If node is right child of parent, and itself has no right child, then succesor to this 
                            node is successor of its parent, excluding this node 
        """
        succ = None
        if self.hasRightChild():
            succ = self.rightChild.findMin()
        else: # node has no right child 
            if self.parent: # has a parent and so is a child
                if self.isLeftChild(): 
                    succ = self.parent # successor is the parent
                else: # is right child of a parent
                    self.parent.rightChild = None
                    succ = self.parent.findSuccessor() 
                    self.parent.rightChild = self
        return succ

    def spliceOut(self):
        """ @fn spliceOut
            @brief to remove the successor
            @note  it goes directly to the node we want to splice out and makes the right changes
                    succ.spliceOut() is used in .remove
        """
        if self.isLeaf():
            if self.isLeftChild():
                self.parent.leftChild = None # update parent to not have a child anymore
            else:  # it's a leaf and a right child
                self.parent.rightChild = None # update parent to not have a child anymore
        else: #self.hasAnyChildren():
            if self.hasLeftChild():
                if self.isLeftChild():
                    self.parent.leftChild = self.leftChild
                else: # node is a right child
                    self.parent.rightChild = self.leftChild
                self.leftChild.parent = self.parent 
            else: # has a right child only
                if self.isLeftChild():
                    self.parent.leftChild = self.rightChild
                else:
                    self.parent.rightChild = self.rightChild
                self.rightChild.parent = self.parent

    def findMin(self):
        """ @fn findMin
            @brief minimum valued key in any binary search tree is leftmost child of tree
        """
        current = self
        while current.hasLeftChild():
            current = current.leftChild
        return current

    def __iter__(self):
        """ @fn __iter__
            @brief The standard inorder traversal of a binary tree.
            @details code for an inorder iterator of a binary tree
            @note yield is similar to return in that it returns a value, 
                    however, yield also freezes state of the function so that 
                    next time function is called, it continues executing from the exact point
                    it left off earlier.  
                    Functions that create objects that can be iterated are called generator functions.  
        """
        if self:
            if self.hasLeftChild():
                for elem in self.leftChild:
                    yield elem
            yield self.key
            if self.hasRightChild():
                for elem in self.rightChild:
                    yield elem

            




    