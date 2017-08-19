package com.cqupt.sdw;


/**
 * Created by STONE on 2017/8/18.
 */
public class LinkedList {
    private ListNode head; //指向链表的第一个节点
    private int sizeOfList = 0; //初始化链表长度

    //构造器
    public LinkedList(){
        head = null;
    }

    /**
     * 加入节点到链表头部
     * @param input
     */
    public void addFront(int input){
        ListNode newNode = new ListNode(input);
        newNode.next = head;
        head = newNode;
        sizeOfList++;
    }

    /**
     * 加入节点到链表尾部
     * @param input
     */
    public void addLast(int input){
        ListNode newNode = new ListNode(input);
        if (head == null){
            head = newNode;
            sizeOfList++;
        }else{
            ListNode temp = head;
            while(temp.next != null){
                temp = temp.next;
            }
            temp.next = newNode;
            sizeOfList++;
        }
    }

    /**
     * 在index位置上加入新的节点
     * @param input
     * @param index
     */
    public void add(int input,int index) {
        if (index < 0 || index > sizeOfList) {
            throw new IndexOutOfBoundsException("索引越界");
        }
        if (index == 0) {
            addFront(input);
        }
        ListNode temp = head;
        for (int i = 0; i < index -1 ;i++){
            temp  = temp.next;
        }
        ListNode newNode = new ListNode(input);
        newNode.next = temp.next;
        temp.next = newNode;
        sizeOfList++;
    }

    /**
     * 移除链表第一个节点
     */
    public void removeFront(){
        if (isEmpty()){
            return;
        }
        head = head.next;
        sizeOfList--;
    }

    /**
     * 移除链表最后一个节点
     */
    public void removeLast(){
        if (isEmpty()){
            return;
        }
        if (head.next == null){
            head = null;

        }else {
            ListNode temp = head;
            while(temp.next.next != null){
                temp = temp.next;
            }
            temp.next = null;
        }
        sizeOfList--;
    }

    /**
     * 移除链表index位置的节点
     * @param index
     */
    public void remove(int index){
        if (index < 0 || index > sizeOfList) {
            throw new IndexOutOfBoundsException("索引越界");
        }
        if (isEmpty()){
            return;
        }
        if (index > 0){
            ListNode temp = head;
            for (int i =0; i< index-1; i++){
                temp = temp.next;
            }
            temp.next = temp.next.next;
        }
        sizeOfList--;
    }

    /**
     * 链表中查询input，返回索引，为查到则返回-1
     * @param input
     * @return
     */
    public int search(int input){
        ListNode temp = head;
        for (int i = 0; i< sizeOfList; i++){
            if (temp.val == input){
                return i;
            }else{
                temp = temp.next;
            }
        }
        return -1;
    }

    public void delete(){
        head = null;
        sizeOfList = 0;
    }

    /**
     * 链表长度
     * @return
     */
    public int size() {
        return sizeOfList;
    }

    /**
     * 链表是否为空
     * @return
     */
    public boolean isEmpty() {
        return sizeOfList == 0;
    }

    /**
     * 打印链表
     */
    public void print(){
        ListNode temp = head;
        while (temp != null){
            System.out.print(temp.val + " ");
            temp = temp.next;
        }
        System.out.println();
    }

    /**
     * 对链表进行排序，返回头节点
     * @return
     */
    public ListNode orderList(){
        ListNode temp = head;
        ListNode nextNode = null;
        while (temp.next != null){
            nextNode = temp.next;
            while (nextNode != null){
                if (temp.val > nextNode.val){   //每次把最小值放到前面
                    int t = temp.val;
                    temp.val = nextNode.val;
                    nextNode.val = t;
                }
                nextNode = nextNode.next;
            }
            temp = temp.next;
        }
        return head;
    }

    /**
     * 删除链表中含重复数据的节点
     * 双层遍历，时间复杂度高
     */
    public void  deleteDuplicate(){
        ListNode temp = head;
        while(temp != null){
            ListNode p = temp;
            while(p.next != null){
                if (temp.val == p.next.val){
                    p.next = p.next.next;
                }else{
                    p = p.next;
                }
            }
            temp = temp.next;
        }
    }

    /**
     * 查找链表倒数第k个节点
     * 思路：维护两个节点，前一个节点比另一个前移k-1步
     * @param k
     * @return
     */
    public ListNode findElement(int k){
        if (k < 1 || k > this.size()){
            return null;
        }
        ListNode p1 = head;
        ListNode p2 = head;
        for (int i = 0; i< k-1;i++){  //前移k-1步
            p1 = p1.next;
        }
        while(p1 != null){
            p1 = p1.next;
            p2 = p2.next;
        }
        return p2;
    }

    /**
     * 链表反转,非递归
     */
    public void reverseList(){
        ListNode newHead = head;
        ListNode temp = head;
        ListNode pre = null;
        while(temp != null){
            ListNode nextNode = temp.next;
            if (nextNode == null){
                newHead = temp;
            }
            temp.next = pre;
            pre = temp;
            temp = nextNode;
        }
        this.head = newHead;
    }

    /**
     * 从尾到头输出链表，递归
     * @param head
     */
    public void printListReverse(ListNode head){
        if (head != null){
            printListReverse(head.next);
            System.out.print(head.val + " ");
        }
    }

    /**
     * 查找链表的中间节点
     * 思路：维护两个节点，快节点每次走两步，慢节点每次走一步；快节点到大尾部时，慢节点在中间
     * @param head
     * @return
     */
    public ListNode searchMid(ListNode head){
        ListNode p1 = head;
        ListNode p2 = head;
        while(p1 != null && p1.next != null && p1.next.next != null) {
            p1 = p1.next.next;
            p2 = p2.next;
        }
        return p2;
    }

    public static void main(String[] args) {
        LinkedList list = new LinkedList();

        list.addFront(1);
        list.addFront(2);
        list.addLast(3);
        list.add(5,1);
        list.print();
        System.out.println("链表长度: " + list.size());

        list.removeFront();
        list.removeLast();
        list.remove(1);
        list.print();
        System.out.println("链表长度： "+ list.sizeOfList);
        list.delete();
        System.out.println("链表清空后：" );
        System.out.println("链表是否为空：" + list.isEmpty());
        list.print();

        list.addFront(1);
        list.addFront(3);
        list.addFront(2);
        int index = list.search(1);
        System.out.println("index = " + index);
        list.print();
        list.orderList();
        System.out.println("排序后：" );
        list.print();
        System.out.println("反转后：");
        list.reverseList();
        list.print();
    }
}
