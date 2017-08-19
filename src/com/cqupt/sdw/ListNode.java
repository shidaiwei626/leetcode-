package com.cqupt.sdw;

/**
 * Created by STONE on 2017/7/21.
 * 定义链表节点
 */
public class ListNode {
     int val; //数值域
     ListNode next;  //指针域
     ListNode(int val) {    //构造方法
         this.val = val;
         this.next = null;
     }

    @Override
    public String toString() {
        return "ListNode{" +
                "val=" + val +
                '}';
    }
}
