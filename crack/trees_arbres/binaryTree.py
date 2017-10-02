"""
	@file   : binaryTree.py
	@brief  : (Binary) Tree 
    @details : binaryTree 
	@author : Ernest Yeung	ernestyalumni@gmail.com
	@date   : 20170921
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
class binaryTree(object):
    """ @class binaryTree
    """
    def __init__(self,key=None):
        self.key = key
        self.l = None   # left child 
        self.r = None   # right child 

    def insertl(self,newkey):
        if self.l is None:
            self.l = binaryTree(newkey)  
        else: # push down left child to the left; we'll have to do a swap
            l = self.l
            self.l = binaryTree(newkey)
            self.l.l = l # push down previous left child to be left child of new left child
    
    def insertr(self,newkey):
        if self.r is None:
            self.r = binaryTree(newkey)  
        else: # push down right child to the right; we'll have to do a swap
            r = self.r
            self.r = binaryTree(newkey)
            self.r.r = r # push down previous left child to be left child of new left child
    
    def isLeaf(self):
        return self.l is None and self.r is None

    """
    cf. http://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/  
    depth 1st traversals 
    inorder   (left, root, right): 4 2 5 1 3 
    preorder  (root, left, right): 1 2 4 5 3
    postorder (left, right, root): 4 5 2 3 1 
    """

    def inorder(self):
        # 1st recur on left child
        if self.l:
            self.l.inorder()
        # then print the data of node
        print(self.key)

        # now recur on right child
        if self.r:
            self.r.inorder()

    def postorder(self):
        # first recur on left child
        if self.l:
            self.l.inorder()
        # the recur on right child
        if self.r:
            self.r.inorder()
        # now print the data of node
        print(self.key)

    def preorder(self):
        # 1st print the data of node 
        print(self.key)

        # then recur on left child
        if self.l:
            self.l.inorder()

        # finally recur on right child
        if self.r:
            self.r.inorder()
