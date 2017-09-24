package com.cqupt.sdw;

import sun.reflect.generics.tree.Tree;

import java.util.*;
import java.util.LinkedList;

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
     * BST插入数据,构建BST时插入
     * @param data
     */
    public void insert(int data){
        TreeNode newNode =  new TreeNode(data);
       if (rootNode == null){
           rootNode = newNode;
           return;
       }else{
           TreeNode current = rootNode;
           TreeNode parent; //新节点的父节点
           while (true){
               parent = current;
               if (data <= current.val){
                   current = current.left;
                   if (current == null){
                       parent.left = newNode;
                       return;
                   }else {

                   }
               }else {
                   current = current.right;
                   if (current == null){
                       parent.right = newNode;
                       return;
                   }
               }
           }
       }
    }

    /**
     * 将数值插入构建二叉树
     * @param arr
     */
    public void buildBST(int[] arr){
        for (int i = 0; i < arr.length; i++){
            insert(arr[i]);
        }
    }

    /**
     * BST 中查找数据，并返回该节点，否则为null
     * @param data
     * @return
     */
    public TreeNode search(int data){
        TreeNode current = rootNode;
        while(current != null){
            if (current.val == data){
                return current;
            }else if (data <= current.val){
                current = current.left;
            }else {
                current = current.right;
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

    /**
     * 递归 中序遍历打印：左子树--根--右子树
     * 输出从小到达排序
     * @param rootNode
     */
    public void printInOrder(TreeNode rootNode) {
        if (rootNode != null) {
            printInOrder(rootNode.left);
            System.out.print(rootNode.val + " ");
            printInOrder(rootNode.right);
        }
    }

    /**
     * 递归 前序遍历打印： 根--左子树--右子树
     * @param rootNode
     */
    public void printPreOrder(TreeNode rootNode){
        if (rootNode != null){
            System.out.print(rootNode.val + " ");
            printPreOrder(rootNode.left);
            printPreOrder(rootNode.right);
        }
    }

    /**
     * 递归 后序遍历打印： 左子树--右子树--根
     * @param rootNode
     */
    public void printPostPrint(TreeNode rootNode){
        if (rootNode != null){
            printPostPrint(rootNode.left);
            printPostPrint(rootNode.right);
            System.out.print(rootNode.val + " ");
        }
    }

    /**
     * 层序遍历二叉树，按层打印节点
     * 思路：先将根节点放在队列中，然后每次都从队列中取出一个节点并打印该节点的值，
     *       若该节点有子节点，则将它的子节点放入队列尾，直到队列为空。
     */
    public void layerPrint(){
         if (this.rootNode == null)
             return;
        java.util.Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.add(this.rootNode);
        while (!q.isEmpty()){
            TreeNode node = q.poll();
            System.out.print(node.val);
            System.out.print(" ");
            if (node.left != null){
                q.add(node.left);
            }
            if (node.right != null){
                q.add(node.right);
            }
        }
    }

    /**
     * 由二叉树的前序遍历和中序遍历构造该二叉树
     * 思路：1.确定树的根节点。前序遍历的第一个节点即为根节点
     *       2.求解树的子树。根据上一步的根节点，在中序遍历中该节点左边为左子树，右边为右子树
     *       3.对根节点的左、右子树递归求解步骤1、2
     * @param preOrder
     * @param inOrder
     */
    public void initTree(int[] preOrder, int[] inOrder){
        rootNode  = initTree(preOrder, 0, preOrder.length - 1, inOrder, 0, inOrder.length - 1 );
    }

    public TreeNode initTree(int[] preOrder, int start1, int end1, int[] inOrder, int start2, int end2){
        if (start1 > end1 || start2 > end2){
            return null;
        }
        int rootVal = preOrder[start1];
        TreeNode rootNode = new TreeNode(rootVal);
        //找到根节点的位置
        int rootIndex = findIndexInArray(inOrder,rootVal,start2,end2);
        int offSet = rootIndex - start2 -1;//左子树节点个数
        //构建左子树
        TreeNode left = initTree(preOrder, start1 + 1, start1 + 1 + offSet, inOrder, start2, start2 + offSet);
        //构建右子树
        TreeNode right = initTree(preOrder, start1 + offSet + 2, end1, inOrder, rootIndex + 1, end2);
        rootNode.left = left;
        rootNode.right = right;
        return rootNode;
    }

    /**
     * 求解根节点索引
     * @param a
     * @param x
     * @param start
     * @param end
     * @return
     */
    public int findIndexInArray(int[] a, int x, int start, int end){
        for (int i = start; i <= end; i++){
            if (a[i] == x){
                return i;
            }
        }
        return -1;
    }


    public static void main(String[] args) {
        BinarySearchTree tree = new BinarySearchTree();
        int[] arr = {10,9,3,12,14,7,6,11,1,2};
        tree.buildBST(arr); //构建二叉树

        System.out.println("层序遍历：");
        tree.layerPrint();
        System.out.println(" ");

        System.out.println("中序遍历：");
        tree.printInOrder(tree.rootNode);
        System.out.println(" ");

        System.out.println("前序遍历：");
        tree.printPreOrder(tree.rootNode);
        System.out.println(" ");

        System.out.println("后序遍历：");
        tree.printPostPrint(tree.rootNode);
        System.out.println(" ");

        //删除节点
        //tree.delete(9);

        //查找节点
        if (tree.search(4) != null){
            System.out.println("查找到该节点");
        }else{
            System.out.println("该节点不存在");
        }

        //由前序遍历和中序遍历构建二叉树
        BinarySearchTree t = new BinarySearchTree();
        int[] pre = {10,9,3,1,2,7,6,12,11,14};
        int[] in = {1,2,3,6,7,9,10,11,12,14};
        t.initTree(pre,in);
        System.out.println("后续遍历：");
        t.printPostPrint(t.rootNode);
    }
}
