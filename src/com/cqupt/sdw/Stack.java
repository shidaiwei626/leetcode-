package com.cqupt.sdw;

/**
 * Created by STONE on 2017/8/18.
 * 1.利用链表实现栈
 * 2.链表最后一个元素为栈顶
 * 3.入栈：链表尾部加入元素
 * 4.出栈：链表尾部删除元素
 */
public class Stack {

    private LinkedList stack;

    //构造器
    public Stack() {
        stack = new LinkedList();
    }

    //入栈
    public void push(int data) {
        stack.addLast(data);
    }

    //出栈
    public void pop()  {
        stack.removeLast();
    }

    public void printStack(){
        stack.print();
    }

    public static void main(String[] args) {
        Stack obj = new Stack();

        //入栈10个数字
        for (int i = 1; i <= 10; i++) {
            obj.push(i);
            System.out.println("Pushed " + i);
        }
        obj.printStack();

        //出栈
        obj.pop();
        obj.printStack();
    }
}
