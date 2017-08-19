package com.cqupt.sdw;

/**
 * Created by STONE on 2017/8/18.
 * 1.链表实现队列
 * 2.队头用表头表示，队尾用表尾表示
 * 3.入队：链表尾部加入元素
 * 4.出队：链表头部删除元素
 */
public class Queue {
    private LinkedList queue;

    //构造器
    public Queue() {
        queue = new LinkedList();
    }

    //入队
    public void push(int data) {
        queue.addLast(data);
    }

    //出队
    public void pop()  {
        queue.removeFront();
    }

    public void printQueue(){
        queue.print();
    }

    public static void main(String[] args) {
        Queue obj = new Queue();

        //入队10个数字
        for (int i = 1; i <= 10; i++) {
            obj.push(i);
            System.out.println("Pushed " + i);
        }
        obj.printQueue();

        //出栈
        obj.pop();
        obj.printQueue();

    }
}
