package com.cqupt.sdw;

import java.util.NoSuchElementException;

/**
 * Created by Shidw on 2017/8/19.
 * 二叉搜索数，对任一节点，左子树所有值小于该节点的值，右子树所有值大于该节点的值
 */
public class BinarySearchTree {
    private TreeNode rootNode; //根节点

    /**
     * 构造函数
     */
    public BinarySearchTree(){
        rootNode = null;
    }

    public BinarySearchTree(int data){
        rootNode = new TreeNode(data);
    }

    /**
     * BST插入数据
     * @param data
     */
    public void insert(int data){
       if (rootNode == null){
           rootNode = new TreeNode(data);
           return;
       }
       TreeNode newNode = new TreeNode(data);
       TreeNode iterator = rootNode;
       TreeNode parent = null; //新节点的父节点
        while (iterator != null){
            parent = iterator;
            if (data <= iterator.val){
                iterator = iterator.left;
            }else {
                iterator = iterator.right;
            }
        }
        if (data <= parent.val){
            parent.left = newNode;
        }else {
            parent.right = newNode;
        }
    }

//    //将指定的值加入到二叉树中适当的节点
//    public void Add_Node_To_Tree(int val){
//        TreeNode currentNode = rootNode;
//        if (rootNode == null){
//            rootNode = new TreeNode(val);
//            return;
//        }
//        //建立二叉树
//        while (true){
//            if (val < currentNode.val) {
//                if (currentNode.left == null){
//                    currentNode.left = new TreeNode(val);
//                    return;
//                }else {
//                    currentNode = currentNode.left;
//                }
//            }else {
//                if (currentNode.right == null){
//                    currentNode.right = new TreeNode(val);
//                }else {
//                    currentNode = currentNode.right;
//                }
//            }
//        }
//    }

    /**
     * BST 中查找数据，并返回该节点，否则为null
     * @param data
     * @return
     */
    public TreeNode search(int data){
        TreeNode iterator = rootNode;
        while(iterator != null){
            if (iterator.val == data){
                return iterator;
            }else if (data <= iterator.val){
                iterator = iterator.left;
            }else {
                iterator = iterator.right;
            }
        }
        return null;
    }

    /**
     *  Finds the parent of the node to be deleted
     * @param data
     * @return
     */
    public boolean delete(int data){
        TreeNode iterator = rootNode;
        TreeNode parent = null;
        while (iterator != null) {
            if (data == iterator.val) {
                return deleteNode(data, parent);
            } else  {
                parent = iterator;
                if (data <= iterator.val)
                    iterator = iterator.left;
                else
                    iterator = iterator.right;
            }
        }
       return  false;
    }

    private boolean deleteNode(int data, TreeNode parent){
        TreeNode child;
        boolean position = false;	// Indicates position of child wrt to parent, true is left child, false is right child
        if (data <= parent.val) {
            child = parent.left;
            position = true;
        }
        else
            child = parent.right;

        if (child.left == child.right) {	// Condition for leaf node
            child = null;
            if (position)
                parent.left = null;
            else
                parent.right = null;
            return true;
        } else if (child.right == null) {	// Condition for non-leaf with no right sub-tree
            if (position)
                parent.left = child.left;
            else
                parent.right = child.left;
            child.left = null;
            child = null;
            return true;
        } else if (child.left == null) {	// Condition for non-leaf with no left sub-tree
            if (position)
                parent.left = child.right;
            else
                parent.right = child.right;
            child.right = null;
            child = null;
            return true;
        }
        else {	// Conditon when Node has both subtree avaliable
            TreeNode iterator = child.right;
            TreeNode parentOfIterator = null;
            while(iterator.left != null) {	// Finding the leftmost child of right sub-tree
                parentOfIterator = iterator;
                iterator = iterator.left;
            }
            child.val = iterator.val;
            parentOfIterator.left = null;
            iterator = null;
            return true;
        }
    }

    public void printInOrder() {	// Function to call inorder printing using root
        if (rootNode == null)
            System.out.println("Cannot print! BST is empty");
        print(rootNode);
        System.out.println(" ");
    }

    /**
     *递归 中序遍历打印：左子树--根--右子树
     * @param node
     */
    private void print(TreeNode node) {
        if (node != null) {
            print(node.left);
            System.out.print(node.val + " ");
            print(node.right);
        }
    }

    /**
     * 递归 前序遍历打印： 根--左子树--右子树
     * @param node
     */
    private void prePrint(TreeNode node){
        if (node != null){
            System.out.println(node.val + " ");
            prePrint(node.left);
            prePrint(node.right);
        }
    }

    /**
     * 递归 后序遍历打印： 左子树--右子树--根
     * @param node
     */
    private void postPrint(TreeNode node){
        if (node != null){
            postPrint(node.left);
            postPrint(node.right);
            System.out.println(node.val + " ");
        }
    }

    public static void main(String[] args) {

        // Created an empty tree
        BinarySearchTree tree = new BinarySearchTree();
        // Adding a few test entries
        tree.insert(10);
        tree.insert(9);
        tree.insert(3);
        tree.insert(12);
        tree.insert(14);
        tree.insert(7);
        tree.insert(6);
        tree.insert(11);
        tree.insert(1);
        tree.insert(2);
        // Test printing
        tree.printInOrder();
        // Deleting a valid node
        tree.delete(9);
        // Print again
        tree.printInOrder();
        // Searching an invalid node, same can be tested for delete as both use same logic
        // but with a slight different approach to find the node
        try {
            tree.search(4);
            System.out.println("Node was found successfully.");
        } catch (Exception e) {
            System.out.println("Invalid Search");
        }
        try {
            tree.delete(9);
        } catch (Exception  e) {
            System.out.println("Cannot delete, Node not present.");
        }
    }
}
