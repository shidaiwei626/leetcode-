package com.cqupt.sdw;

/**
 * Created by STONE on 2017/7/21.
 * 定义单链表
 */
public class ListNode {
     int val; //数值域
     ListNode next;  //指针域
     ListNode(int x) {    //构造方法
         val = x;
     }

    @Override
    public String toString() {
        return "ListNode{" +
                "val=" + val +
                '}';
    }

}
