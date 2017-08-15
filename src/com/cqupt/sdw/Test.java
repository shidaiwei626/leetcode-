package com.cqupt.sdw;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

/**
 * Created by STONE on 2017/7/13.
 */
public class Test {
    public static void main(String[] args){
        int[] a ={1,2,3};
        Solution so = new Solution();
        System.out.println(Arrays.toString(so.twoSum(a, 4)));
        System.out.println(so.reverse(-123));

        String b = "mississippi", c="issi";
        int index = so.strStr(b,c);
        System.out.println(index);
        String e ="11",f = "11";
        System.out.println(so.addBinary2(e,f));
//        char a2 = '1',b2= '0';
//        int c2 = a2-b2;
//        System.out.println(c2);
        int[] g = new int[2];
        System.out.println(Arrays.toString(g));

        int h = 9;
        System.out.println(so.mySqrt(h));

        int[][] j = {{0,1},{2,3}};
       // so.setZeroes(j);
        System.out.println(so.searchMatrix(j,2));

        int[] t= {3,4,1,0,4};
        System.out.println(so.canJump2(t));

        String s = "a";
        System.out.println(so.longestPalindrome(s));

        int[] nums = {-1,-2,-3,0,-1,1,2,3,4};
        System.out.println("三数之和为0："+ so.threeSum(nums));

        String  digits ="23";
        System.out.println("字母组合：" + so.letterCombinations(digits));
        System.out.println(so.rob2(t));

        List<Interval> intervals =new ArrayList<Interval>(){{add(new Interval(1,3));
        add(new Interval(2,6));add(new Interval(8,10));add(new Interval(15,18));}} ;
        List<Interval> list = so.merge(intervals);
        for(int i = 0;i< list.size();i++){
            System.out.println(list.get(i));
      }

       int[] prices = {1,3,4,7,9};
        System.out.println(so.maxProfit(prices));
        System.out.println(so.binarySearch(prices,7));

        String ss = "the sky is blue";
        System.out.println(so.reverseWords(ss));

        int[][] ba = {{1,2,3},{4,5,6},{7,8,9}};
        so.rotate(ba);
        for(int i = 0;i<ba.length;i++){  //输出二维数组
            System.out.println(Arrays.toString(ba[i]));
        }

        int[][] matrix = {{1,2,3},{4,5,6},{7,8,9}};
        System.out.println(so.searchMatrix2(matrix,3));


        int[] m = {1,1,1,1,2,3,3,3,4,5};
        System.out.println(so.removeDuplicates(m));

        ListNode listnode = new ListNode(1);
        listnode.next = new ListNode(3);

        System.out.println(so.deleteDuplicates(listnode).toString());

        int[] n = {1,1,2};
        System.out.println(so.removeDuplicate(n));

        int[] nn = {3,4,-1,1};
        System.out.println(so.firstMissingPositive(nn));

        int[] nums1 = new int[6];
        nums1[0] = 1;
        nums1[1] = 2;
        nums1[2] = 3;
        int[] nums2 = new int[]{4,5,6};
        so.merge(nums1,3,nums2,3);
        System.out.println(Arrays.toString(nums1));

        String s1 = "001",s2 ="010";
        System.out.println("两个二进制字符串的积：" + so.multiply(s1,s2));

//        String jsonString = "{searchId:1,results:[{roomId:1,rfids:[1,2,3,4]},{roomId:2,rfids:[9,8,7,6]}]}";
//        JSONObject json = JSON.parseObject(jsonString);
//        JSONArray jsonArray = (JSONArray) json.get("results");
//
//        for (int i = 0 ;i<jsonArray.size();i++){
//           // System.out.println(jsonArray.get(i));
//            JSONObject ii = (JSONObject) jsonArray.get(i);
//           // System.out.println(ii.get("rfids").toString());
//            ArrayList<Integer> rfidList = (ArrayList) ii.get("rfids");
//            for (Integer rfid :rfidList){
//                System.out.println(rfid);
//            }
//        }
        //System.out.println(json.getString("results"));

        //String searchResults = json.
      //  JSONObject json2 = JSON.parseObject(searchResults);
     //   JSONArray json2 = JSON.parseArray(searchResults);
      //  System.out.println(json.getString("results"));
       // System.out.println(json2.toString());
        //System.out.println(json2.getString("rfids"));


        String jsonString = "{\"searchId\":\"5\",\"results\":[{\"roomId\":\"305\",\"rfids\":[\"005\",\"520\"]},{\"roomId\":\"303\",\"rfids\":[\"53\",\"43\"]},{\"roomId\":\"302\",\"rfids\":[\"30\",\"60\"]}]}";
        JSONObject json = JSON.parseObject(jsonString);
        JSONArray jsonArray = (JSONArray) json.get("results"); //获得json数组
        System.out.println(jsonArray);
        for (int i = 0; i < jsonArray.size(); i++) {
            JSONObject roomResult = (JSONObject) jsonArray.get(i);
            //   String roomId = (String) roomResult.get("roomId");
            JSONArray jsonArray2 = (JSONArray) roomResult.get("rfids");
            for (int p = 0; p < jsonArray2.size(); p++) {
                System.out.println((String) jsonArray2.get(p));
            }
        }
//            String[] rfidList = roomResult.get("rfids").toString().split(",");
//            for (String rfid : rfidList) {
//                System.out.println(rfid);
//            }
           // System.out.println(roomResult.get("rfids"));

            int[] arr = {3,44,38,5,47,88};
//            so.bubbleSort(arr);
//            System.out.println(Arrays.toString(arr));

            int[][] arr2 = {{1,4,5},{2,6,8},{4,7,9}};
            System.out.println(so.search(arr2,7));
//            so.insertSort(arr);
//            System.out.println(Arrays.toString(arr));
//            so.selectSort(arr);

                so.mergeSort(arr,1,arr.length-1);
                System.out.println(Arrays.toString(arr));

        System.out.println( so.reverse(-123456));

        System.out.println(so.strStr("agcd","cdw"));

        int[][] zeros = {{1,0,4},{1,2,3}};
        so.setZeroes(zeros);
        for(int i = 0;i< zeros.length;i++){  //输出二维数组
            System.out.println(Arrays.toString(zeros[i]));
        }

        System.out.println(so.isPrimeNumber(101));

        System.out.println(so.dividePrimeFactor(101));

        System.out.println(so.replaceBlank("abc de f"));

        System.out.println(so.sum(1,3));

        System.out.println(so.isPerfectNum(6));

        System.out.println(so.sum(new int[]{1,2,3,4}));

        System.out.println(so.answer());

        System.out.println(so.isLeapYear(2004));

        System.out.println(so.dayOfYear(new String[]{"2016","3","3"}));

        System.out.println(Arrays.toString(so.threeSort(new int[]{5,4,3})));

        int[] sour = new int[]{2,3,4,10,2,3};
        so.countingSort(sour);
        System.out.println(Arrays.toString(sour));

        System.out.println(so.getClass().getName()); //获取类名
        System.out.println(so.getClass().getSuperclass().getName()); //获取父类名

        System.out.println(so.mySqrt(8));

        String cc1 = "hello";
        String cc2 = "石代伟";
        System.out.println("cc1长度：" + cc1.length() + "字节数：" + cc1.getBytes().length);
        System.out.println("cc2长度：" + cc2.length() + "字节数：" + cc2.getBytes().length);

        //StringTokenizer类，分割字符串
        StringTokenizer st = new StringTokenizer("Welcome to China!");
        while (st.hasMoreTokens()){
            System.out.println(st.nextToken());
        }

    }
    static{
        System.out.println("ahahhahaahha");   //静态代码块在main方法之前执行，当类被加载时就会被调用。
    }
}
