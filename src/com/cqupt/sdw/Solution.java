package com.cqupt.sdw;

import java.util.*;
import java.lang.*;

/**
 * Created by STONE on 2017/7/13.
 */
public class Solution {

    /**
     * 1.Two Sum  给定数组和目标值，求数组中两数之和为目标值的索引
     * 如 nums = [2, 7, 11, 15], target = 9,
     * nums[0] + nums[1] = 2 + 7 = 9, return [0, 1].
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum(int[] nums, int target) {
        int[] result = new int[2];
        Map<Integer, Integer> map = new HashMap();
        int nu = nums.length;
        for (int i = 0; i < nu; i++) {
            if (map.containsKey(target - nums[i])) {
                result[1] = i;
                result[0] = map.get(target - nums[i]);
                return result;
            } else {
                map.put(nums[i], i);
            }
        }
        return result;
    }

    /**
     * 7. Reverse Integer,输入为32bit 有符号整数
     * 翻转数字，如123变为321，-123变为-321
     * @param x
     * @return
     */
    public int reverse(int x) {
        long result = 0;
        int temp = Math.abs(x);
        while (temp > 0) {
            result = result * 10;
            result = result + temp % 10;
            if (result > Integer.MAX_VALUE) {
                return 0; //溢出时返回0
            }
            temp = temp / 10;
        }
        return (int) (x > 0 ? result : -result);
    }

    /**
     * 28.Implement strStr()函数，返回needle在haystack中第一次出现的位置，如果没找到，则返回-1
     * @param haystack
     * @param needle
     * @return
     */
    public int strStr(String haystack, String needle) {
        int n1 = haystack.length();
        int n2 = needle.length();
        int index = -1;
        if (n1 >= n2) {
            for (int i = 0; i <= n1 - n2; i++) {
                if (needle.equals(haystack.substring(i, n2 + i))) {
                    index = i;
                    break;
                }
            }
        }
        return index;
    }

    /**
     * /给定两个二进制字符串返回其和
     * For example,a = "11",b = "1",Return "100".
     * @param a
     * @param b
     * @return
     */
    public String addBinary(String a, String b) {
        int i = a.length() - 1;
        int j = b.length() - 1;
        int carry = 0; //进位标记
        char[] achar = a.toCharArray();  //将String转为char数组
        char[] bchar = b.toCharArray();
        // 结果数组
        char[] resultchar = new char[Math.max(achar.length, bchar.length) + 1];
        // 标记结果数组位置
        int resultIndex = 0;
        while (true) {
            if (i < 0 && j < 0 && carry == 0) break;
            int aflag = 0;
            int bflag = 0;
            if (i >= 0) aflag = achar[i] - '0';
            if (j >= 0) bflag = bchar[j] - '0';
            if (aflag + bflag + carry > 1) {
                resultchar[resultIndex] = (char) ('0' + aflag + bflag + carry - 2);
                carry = 1;
            } else {
                resultchar[resultIndex] = (char) ('0' + aflag + bflag + carry);
                carry = 0;
            }
            resultIndex++;
            i--;
            j--;
        }
        String result = new String(resultchar, 0, resultIndex);
        StringBuffer buffer = new StringBuffer(result);
        result = buffer.reverse().toString();
        return result;
    }

    public String addBinary2(String a, String b) {
        int x = Integer.valueOf(a, 2);
        int y = Integer.valueOf(b, 2);
        int z = x + y;
        String result = Integer.toBinaryString(z);
        return result;
    }

    /**
     * 70.Climbing Stairs.给定梯子数n,每次能爬一节或者两节，返回爬到最高处的方法数
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        // 斐波那契数列问题,动态规划解决
        if (n <= 1) return 1;
        int[] dp = new int[n];
        dp[0] = 1;
        dp[1] = 2;
        for (int i = 2; i < n; ++i) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n - 1];
    }

    /**
     * 69. Sqrt(x),Implement int sqrt(int x).Compute and return the square root of x
     * 二分法计算平方根
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        if (x <= 1) {
            return x;
        }
        int mid = 0;
        int begin = 1;
        int end = x;
        while (begin <= end) {
            mid = (begin + end)/ 2;
            if (mid == x / mid) {
                return mid;
            } else if (mid < x / mid) {
                begin = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        //结束条件end一定<begin，所以返回end,如8，返回2
        return end;
    }

    /**
     * 73. Set Matrix Zeroes
     * 遍历二维矩阵，使元素为0的行列数据置为0,在遍历第一行和第一列的值，最后遍历首元素
     * @param matrix
     */
    public void setZeroes(int[][] matrix) {
        int r = matrix.length;
        int c = matrix[0].length;
        //遍历第一行和第一列，查看是否含有0
        boolean firstRow = false, firstCol = false;
        for (int i = 0; i < r; i++) {
            if (matrix[i][0] == 0) {
                firstCol = true;
            }
        }
        for (int j = 0; j < c; j++) {
            if (matrix[0][j] == 0) {
                firstRow = true;
            }
        }

        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;   //将元素为0的行列存入对应的第0行，第0列
                    matrix[i][0] = 0;
                }
            }
        }

        //将对应的行和列置0
        for (int i = 1; i < r; i++) {
            if (matrix[i][0] == 0) {
                for (int j = 1; j < c; j++)
                    matrix[i][j] = 0;
            }
        }
        for (int j = 1; j < c; j++) {
            if (matrix[0][j] == 0) {
                for (int i = 1; i < r; i++)
                    matrix[i][j] = 0;
            }
        }

        if (firstRow) {  //首元素最后单独判断，否则会影响第一行和第一列中的元素
            for (int j = 1; j < c; j++) {
                matrix[0][j] = 0;
            }
        }
        if (firstCol){
           for (int i = 1; i < r; i++){
                matrix[i][0] = 0;
           }
        }
    }

    /**
     * 74.Search a 2D Matrix,
     * 在二维矩阵中查找目标值，返回状态,输入矩阵元素从小到大
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int r = matrix.length;
        if (r == 0) return false;
        int c = matrix[0].length;
        boolean state = false;
        for (int i = 0; i < r; i++) {
            if (matrix[i][0] == target) {
                state = true;
            } else if (matrix[i][0] < target) {
                continue;
            } else {
                for (int j = 1; j < c; j++) {
                    if (matrix[i - 1][j] == target) {
                        state = true;
                    }
                }
            }
        }
        return state;
    }

    public boolean searchMatrix2(int[][] matrix, int target) {
        //法2:二分查找法
        int r = matrix.length;
        if (r == 0) return false;
        int c = matrix[0].length;
        int[] newArray = new int[r * c];
        int k = 0;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                newArray[k] = matrix[i][j];
                k++;
            }
        }
        int left = 0, right = newArray.length - 1, mid = 0;
        while (left <= right) {
            mid = (left + right) / 2;
            if (newArray[mid] == target) {
                return true;
            } else if (newArray[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return false;
    }


    /**
     * 441. Arranging Coins,输出能够搭成的梯子数
     * @param n
     * @return
     */
    public int arrangeCoins(int n) {
        int sum = 0;
        int last = 0;
        if (n == 1) return 1;
        for (int i = 1; i <= n; ) {
            sum = sum + i;
            if (sum < n) {
                i++;
            } else if (sum == n) {
                last =  i;
                break;
            } else {
                last = i - 1;
                break;
            }
        }
        return  last;
    }

    /**
     * 62. Unique Paths
     * 求解从矩阵左上角到右下角的路径数，只能右移和下移
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        //动态规划问题
        //到达每一个网格的路径数等于它上面和左面网格的路径数之和
        int[][] array = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                array[i][j] = 1;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                array[i][j] = array[i - 1][j] + array[i][j - 1];
            }
        }
        return array[m - 1][n - 1];
    }

    /**
     * 64. Minimum Path Sum
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        //动态规划问题
        //到达每一个网格的最小的和等于它上面和左面网格的最小值加上该网格的值
        int m = grid.length;
        int n = grid[0].length;
        int[][] paths = new int[m][n];
        paths[0][0] = grid[0][0];

        for (int i = 1; i < m; ++i) {    //第一列，只能从左到右走
            paths[i][0] = paths[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < n; ++j) {    //第一行，只能从上到下走
            paths[0][j] = paths[0][j - 1] + grid[0][j];
        }

        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                paths[i][j] = Math.min(paths[i - 1][j], paths[i][j - 1])
                        + grid[i][j];
            }
        }
        return paths[m - 1][n - 1];
    }

    /**
     * 55. Jump Game
     * 给定数组，数组元素表示能前进的最大步数，最开始时在第一个元素位置，是否可到达最后一个元素位置，数组中可能含有0元素
     * 贪心算法：每一步都确定能够跳跃的最大距离，如果最后一个元素在这个距离内则表示可达
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        int size = nums.length;
        if (size <= 0) {
            return false;
        }

        int maxJump = -1;
        for (int i = 0; i < size; i++) {
            if (nums[i] > maxJump) {
                maxJump = nums[i];//确定该位置能够跳跃的最大距离
            }
            if (maxJump >= size - i - 1) {
                return true;
            }
            if (maxJump == 0) {
                return false;
            }
            maxJump--; //下一个位置时，相对于该位置能够达到的最大距离
        }
        return false;
    }

    // 法2：{3,2,1,0,4}
    public boolean canJump2(int[] A) {
        if( A== null || A.length == 0)
            return false;
        int reach = 0;  //表示可达到的距离，对应数组元素索引，当大于最大索引时表示可达
        for(int i = 0;i <= reach && i < A.length; i++) {
            reach = Math.max(A[i]+i,reach);
        }
        if(reach < A.length-1)
            return false;
        return true;
    }

    /**
     * 45. Jump Game II (未理解)
     * Jump Game的扩展，区别是这道题不仅要看能不能到达终点，而且要求到达终点的最少步数
     * @param A
     * @return
     */
    public int jump(int[] A) {
        if(A == null || A.length==0)
            return 0;
        int lastReach = 0;
        int reach = 0;
        int step = 0;
        for(int i=0;i <= reach && i<A.length;i++) {
            if(i>lastReach) {
                step++;
                lastReach = reach;
            }
            reach = Math.max(reach,A[i]+i);
        }
        if(reach < A.length-1)
            return 0;
        return step;
    }

    /**
     * 2. Add Two Numbers
     * Input: (2 -> 4 -> 3) + (5 -> 6 -> 4),Output: 7 -> 0 -> 8
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0); //初始化单链表
        int carry = 0; //进位标志
        ListNode p = head;
        while (l1 != null || l2 != null) {
            //两个有一个没到头，就继续
            if (l1 != null) {
                carry += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                carry += l2.val;
                l2 = l2.next;
            }
            p.next = new ListNode(carry % 10);
            carry /= 10;
            p = p.next;
        }
        if (carry > 0) {//处理最后的进位
            p.next = new ListNode(carry);
        }
        return head.next;
    }

    /**
     * 5. Longest Palindromic Substring. 返回输入字符串的最长回文子串。
     * 复杂度太高，题目未通过
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        String longestPalindrome = null;
        int maxlen = 1;
        int nu = s.length();
        for (int i = 0; i < nu; i++) {
            for (int j = i; j <= nu - maxlen - 1; j++) {
                String subStr = s.substring(i, j + maxlen); //每次截取字符串长度
                int len = j + 1 + maxlen - i;
                if (isPalindrome(subStr) && len > maxlen) {
                    longestPalindrome = subStr;
                    maxlen = len;
                }
            }
        }
        return longestPalindrome;
    }

    public static boolean isPalindrome(String s) {
        int n = s.length();
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) != s.charAt(n - i - 1))
                return false;
        }
        return true;
    }

    public String isPalindrome2(String s) {
        if (s.length() == 0 || s.length() == 1) {
            return s;
        }
        boolean[][] dp = new boolean[s.length()][s.length()];
        int i, j;
        for (i = 0; i < s.length(); i++) {
            for (j = 0; j < s.length(); j++) {
                if (i >= j) {
                    dp[i][j] = true; //当i == j 的时候，只有一个字符的字符串; 当 i > j 认为是空串，也是回文
                } else {
                    dp[i][j] = false; //其他情况都初始化成不是回文
                }
            }
        }

        int k;
        int maxLen = 1;
        int rf = 0, rt = 0;
        for (k = 1; k < s.length(); k++) {
            for (i = 0; k + i < s.length(); i++) {
                j = i + k;
                if (s.charAt(i) != s.charAt(j)) //对字符串 s[i....j] 如果 s[i] != s[j] 那么不是回文
                {
                    dp[i][j] = false;
                } else  //如果s[i] == s[j] 回文性质由 s[i+1][j-1] 决定
                {
                    dp[i][j] = dp[i + 1][j - 1];
                    if (dp[i][j]) {
                        if (k + 1 > maxLen) {
                            maxLen = k + 1;
                            rf = i;
                            rt = j;
                        }
                    }
                }
            }
        }
        return s.substring(rf, rt + 1);
    }

    /**
     * 11. Container With Most Water
     * 思路：当左端线段L小于右端线段R时，我们把L右移，
     * 这时舍弃的是L与右端其他线段（R-1, R-2, ...）组成的木桶，因为这些木桶的容积肯定都没有L和R组成的木桶容积大。
     * 两层循环遍历超时，不可取
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        int maxw = 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            int p = (right - left) * Math.min(height[right], height[left]); //计算面积
            if (p > maxw) {
                maxw = p;
            }
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return maxw;
    }

    /**
     * 汉明距：求异或后计算1的个数，java中int类型固定占4个字节
     * @param x
     * @param y
     * @return
     */
    public int hammingDistance(int x, int y) {
        int xor = x ^ y, count = 0;   //异或
        for (int i=0;i<32;i++)
            count += (xor >> i) & 1;  //移位后原值xor不变
        return count;
    }

    /**
     * 返回字符串中最后一个单词的长度 Given s = "Hello World",return 5
     * @param s
     * @return
     */
    public int lengthOfLastWord(String s) {
        String[] sc = s.split(" ");
        int n = sc.length;
        if (n == 1){
            return s.length();
        }else if(n > 1){
            return sc[n-1].length();
        }else{
            return 0;
        }
    }

    /**
     * 268. Missing Number,输入数组大小为0 -- n,Given nums = [0, 1, 3] return 2.
     * 要求线性的时间复杂度和常数级的空间复杂度
     * 解法：等差数列求和后减去数组之和，即为丢失的数
     * @param nums
     * @return
     */
    public int missingNumber(int[] nums) {
        int n = nums.length;
        int sum = nums[0];
        //输入数组可能是单独的0或者1，此时直接执行return语句
        if (n > 1){
            for (int i = 1; i< n; i++ ){
                sum += nums[i];
            }
        }
        return (int)(0.5 * n * (n + 1)) - sum;
    }

    /**
     * Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num
     * calculate the number of 1's in their binary representation and return them as an array.
     * For num = 5 you should return [0,1,1,2,1,2]
     * @param num
     * @return
     */
    public int[] countBits(int num) {
        int[] result = new int[num+1];
        for(int i=0; i<=num; i++){
            result[i] = countEach(i);  //调用下面的方法
        }
        return result;
    }

    public int countEach(int num){   //计算数字二进制中1的个数
        int result = 0;//表示1的个数
        while(num != 0){
            if(num % 2 == 1){
                result++;
            }
            num = num/2;
        }
        return result;
    }

    /**
     * 34. Search for a Range
     *  Given [5, 7, 7, 8, 8, 10] and target value 8,return [3, 4].
     *  思路：二分查找法找到目标值，然后向左右找到目标值的边缘
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        int[] result = { -1, -1 };
        while (left <= right) {
            int mid = (left + right) / 2;   //二分查找法
            if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                result[0] = mid;
                result[1] = mid;
                int i = mid - 1;    //求目标数字的左索引
                while (i >= 0 && nums[i] == target) {
                    result[0] = i;
                    i--;
                }
                i = mid + 1;   //求目标数字的右索引
                while (i < nums.length && nums[i] == target) {
                    result[1] = i;
                    i++;
                }
                break;
            }
        }
        return result;
    }

    /**
     * 3.Longest Substring Without Repeating Characters
     * 求字符串不重复的最长子串，返回其长度.
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        int n = s.length(); //字符串长度
        int len = 0; //计算过程中临时长度
        int maxlen = 0; //最大长度，返回值
        int start = 0; //返回最长子串的起始位置
        Map<Character,Integer> hashmap = new HashMap<Character,Integer>(); //存储字符和索引
        if(s == null || n == 0) {
            return maxlen;
        }
        for(int i = 0;i<n;i++){
            if(!hashmap.containsKey(s.charAt(i))){
                len++;
                if(len > maxlen) maxlen = len;
                hashmap.put(s.charAt(i),i);
            }else{
                int j = hashmap.get(s.charAt(i));
                for(int k = start;k<=j;k++){
                    hashmap.remove(s.charAt(k));  //去除索引j之前的元素,根据key删除
                }
                hashmap.put(s.charAt(i),i);
                start = j +1;
                len = i-j;
            }
        }
        return maxlen;
    }

    /**
     * 15. 3Sum, 找出数组中三个数字之和为0的所有组合，不能包含重复的组合，但三个数字允许出现重复。
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        int nu = nums.length;
        if (nu < 3) return result;
        Arrays.sort(nums);
        for (int i = 0; i < nu; i++) {
            int second = i + 1;
            int third = nu - 1;
            if (i > 0 && nums[i] == nums[i - 1]) continue;  //去除重复结果
            while (second < third) {
                int sum = nums[i] + nums[second] + nums[third];
                if (sum == 0) {
                    List<Integer> list = new ArrayList<Integer>();
                    list.add(nums[i]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    result.add(list);
                    second++;
                    third--;
                    while (nums[third] == nums[third + 1] && second < third) third--;
                    while (nums[second] == nums[second - 1] && second < third) second++; //去除重复结果
                } else if (sum > 0) {
                    third--;
                } else {
                    second++;
                }
            }
        }
        return result;
    }

    /**
     * 17. Letter Combinations of a Phone
     * 思路:回溯法
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        // 建立映射表
        String[] table = {" ", " ", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        List<String> res = new ArrayList<String>();
        helper(res, table, 0, "", digits);
        return res;
    }

    private void helper(List<String> res, String[] table, int idx, String str, String digits) {
        if (idx == digits.length()) {
            // 找到一种结果，加入列表中
            if (str.length() != 0) {
                res.add(str);
                return;
            }
        } else {
            // 找出当前位数字对应可能的字母
            String candidates = table[digits.charAt(idx) - '0'];
            // 对每个可能字母进行搜索
            for (int i = 0; i < candidates.length(); i++) {
                String strC = str + candidates.charAt(i);
                helper(res, table, idx + 1, strC, digits);
            }
        }
    }

    /**
     * 198. House Robber
     * 思路：动态规划，在一列数组中取出一个或多个不相邻数，使其和最大
     * 维护一个一维数组dp，其中dp[i]表示到i位置时不相邻数能形成的最大和
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        int len = nums.length;
        if (len == 0) {
            return 0;
        }
        int[] dp = new int[len];
        for (int i = 0; i < len; i++) {
            if (i == 0) {
                dp[i] = nums[0];
            } else if (i == 1) {
                dp[i] = Math.max(nums[0], nums[1]);
            } else if (i > 1) {
                dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
            }
        }
        return dp[len - 1];
    }

    /**
     * 法2：dp[i][1]表示抢劫当前房子，dp[i][0]表示不抢劫
     * @param num
     * @return
     */
    public int rob2(int[] num) {
        int[][] dp = new int[num.length + 1][2];//为防止下面数组越界，数组长度设为num.length+1
        for (int i = 1; i <= num.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1]);
            dp[i][1] = num[i - 1] + dp[i - 1][0];
        }
        return Math.max(dp[num.length][0], dp[num.length][1]);
    }

    /**
     * 121. Best Time to Buy and Sell Stock, 求最佳收益
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int curmin = prices[0];
        int maxprofit = 0;
        for (int i = 1; i < prices.length; i++) {
            curmin = Math.min(curmin, prices[i]);
            maxprofit = Math.max(maxprofit, prices[i] - curmin);
        }
        return maxprofit;
    }

    /**
     * 56. Merge Intervals,合并给定的区间。如Given [1,3],[2,6],[8,10],[15,18],return [1,6],[8,10],[15,18].
     * @param intervals
     * @return
     */
    public List<Interval> merge(List<Interval> intervals) {
        int nu = intervals.size();
        if (nu <= 1) {
            return intervals;
        }
        Collections.sort(intervals, new Comparator<Interval>() {
            public int compare(Interval o1, Interval o2) {
                return o1.start - o2.start;//起始值升序排序
            }
        });
        List<Interval> mergeList = new ArrayList<Interval>();
        Interval i1 = intervals.get(0);
        //遍历
        for (int i = 0; i < intervals.size(); i++) {
            Interval i2;
            if (i == intervals.size() - 1) {
                i2 = new Interval(Integer.MAX_VALUE, Integer.MAX_VALUE); //如果i到最后，增加一个虚拟最大的区间
            } else {
                i2 = intervals.get(i + 1);
            }
            //合并区间
            if (i2.start >= i1.start && i2.start <= i1.end) {
                i1.end = Math.max(i1.end, i2.end);
            } else {
                mergeList.add(i1); //没有交集，直接添加
                i1 = i2;//i1更迭
            }
        }
        return mergeList;
    }

    /**
     * 33. Search in Rotated Sorted Array nums
     * 如数组4 5 6 7 0 1 2，查找值2。数组不能含重复值
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            //如果左侧是有序的
            if (nums[left] <= nums[mid]) {
                if (nums[mid] > target && nums[left] <= target) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && nums[right] >= target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    /**
     * 49. Group Anagrams
     * given: ["eat", "tea", "tan", "ate", "nat", "bat"],
     * Return:
     * [
     * ["ate", "eat","tea"],
     * ["nat","tan"],
     * ["bat"]
     * ]
     * @param strs
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new ArrayList<List<String>>();
        if (strs.length == 0) {
            return result;
        }
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for (int i = 0; i < strs.length; i++) {
            char[] ch = strs[i].toCharArray();
            Arrays.sort(ch);
            String temp = new String(ch);
            if (map.containsKey(temp)) {
                map.get(temp).add(strs[i]);
            } else {
                map.put(temp, new ArrayList<String>());
                map.get(temp).add(strs[i]);
            }
        }
        for (String str : map.keySet()) {
            result.add(map.get(str));
        }
        return result;
    }

    /**
     * 104. Maximum Depth of Binary Tree
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null)
            return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    /**
     * 二分查找法,输入数据有序,输出为目标值索引，不存在则输出-1
     * @param nums
     * @param target
     * @return
     */
    public int binarySearch(int[] nums, int target) {
        int left = 0, right = nums.length - 1, mid = 0;
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    /**
     * 151. Reverse Words in a String.
     * Given s = "the sky is blue", return "blue is sky the".
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        String[] str = s.split(" ");
        for (int i = 0; i < str.length / 2; i++) {
            String temp = str[i];
            str[i] = str[str.length - 1 - i];
            str[str.length - 1 - i] = temp;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < str.length; i++) {
            sb.append(str[i]);
            sb.append(" ");

        }
        int a = sb.toString().length();
        return sb.toString().substring(0, a - 1);
    }

    /**
     * 48. Rotate Image,旋转数组90度,输入数组为方阵
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        //思路：首先将数组转置，然后交换列数据
        int row = matrix.length;
        int column = row;
        for (int i = 0; i < row; i++) {
            for (int j = i + 1; j < column; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][column - 1 - j];
                matrix[i][column - 1 - j] = temp;
            }
        }
    }

    /**
     * 80. Remove Duplicates from Sorted Array II
     * 返回数组个数，每个元素个数小于等于2，输入数组从小到大排序(可乱序)。
     * 如：[1,1,1,2,2,3] ，返回值为5
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; i++) {
            if (!map.containsKey(nums[i])) {
                map.put(nums[i], 1);
            } else {
                if (map.get(nums[i]) == 1)
                    map.replace(nums[i], 2); //个数替换为2
            }
        }
        int total = 0;
        Iterator it = map.keySet().iterator();
        while (it.hasNext()) {
            Object key = it.next();
            total = total + map.get(key);
        }
        return total;
    }

    /**
     * 法2，输入数组从小到大排序.
     * @param A
     * @return
     */
    public int removeDuplicates2(int[] A) {
        if (A == null || A.length == 0)
            return 0;
        int idx = 0;
        int count = 0;
        for (int i = 0; i < A.length; i++) {
            if (i > 0 && A[i] == A[i - 1]) {
                count++;
                if (count >= 3)
                    continue;
            } else {
                count = 1;
            }
            idx++ ;
        }
        return idx;
    }

    /**
     * 83. Remove Duplicates from Sorted List
     * 去除单链表重复的数据，输入按照从小到大排序
     * 如1->1->2->3->3, return 1->2->3
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null){
            return head;
        }
        ListNode temp = head;
        ListNode cur = head.next;
        while(cur != null ){
            if (cur.val == temp.val){
                temp.next = cur.next; //跳过相同值的节点
                cur = cur.next;
            }else{
                temp = cur;
                cur = cur.next;
            }
        }
        return head;
    }

    /**
     * 82. Remove Duplicates from Sorted List II，
     * 去除单链表中重复的数字
     * Given 1->1->1->2->3, return 2->3
     * @param head
     * @return
     */
    public ListNode deleteDuplicates2(ListNode head) {
        if(head == null)
            return head;
        ListNode pre = new ListNode(0);   //辅助指针
        pre.next = head;
        ListNode temp = pre;
        ListNode cur = head;
        while(cur !=null)
        {
            while(cur.next != null && temp.next.val == cur.next.val)
            {
                cur = cur.next;
            }
            if(temp.next == cur)   //连续两个数不同时
            {
                temp = temp.next;
            } else {    //连续两个数相同时
                temp.next = cur.next;
            }
            cur = cur.next;
        }
        return pre.next;
    }

    /**
     * 137. Single Number II
     * 输入数组中每个元素出现3次，有一个出现一次，找到该元素返回s
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        int n = nums.length;
        int target = 0;
        if( n ==1 || n == 0)
            target = nums[0];
        Arrays.sort(nums);
        if(n>=2){
            if(nums[0] != nums[1]){
                target = nums[0];
            }
            if(nums[n-1] != nums[n-2]){
                target = nums[n-1];
            }
            for(int i=2;i <n-2; i++){
                if(nums[i-1] != nums[i] && nums[i] != nums[i+1]){
                    target = nums[i];
                }
            }
        }
        return target;
    }

    /**
     * 167. Two Sum II - Input array is sorted
     * 输入数组从小到大，无重复，返回两个元素之和等于目标值的索引，从1开始。
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum3(int[] numbers, int target) {
        int[] index = new int[2];
        Map<Integer, Integer> map = new HashMap<Integer,Integer>();
        for(int i = 0; i< numbers.length;i++){
            if(!map.containsKey(target-numbers[i])){
                map.put(numbers[i],i+1);
            }else{
                index[0] = map.get(target-numbers[i]);
                index[1] = i+1;
            }
        }
        return index;
    }

    /**
     * 14. Longest Common Prefix
     * 返回字符串数组最长公共前缀
     * @param strs
     * @return
     */
    public String longestCommonPrefix(String[] strs) {
        StringBuilder strb = new StringBuilder();
        if(strs != null && strs.length > 0){
            Arrays.sort(strs);
            char[] a = strs[0].toCharArray();
            char[] b = strs[strs.length-1].toCharArray();
            for(int i = 0;i < a.length; i++){
                if(b.length >i && b[i] == a[i]){
                    strb.append(a[i]);
                }else{
                    return strb.toString();
                }
            }
        }
        return strb.toString();
    }

    /**
     * 26. Remove Duplicates from Sorted Array
     * 输入有序数组，去除数组中的重复值并返回新数组的长度，不能额外分配空间
     * @param nums
     * @return
     */
    public int removeDuplicate(int[] nums) {
        int len = nums.length;
        int s = 1; //返回值
        if(len == 1) s=1;
        for(int i = 1 ;i< len ;i++){
            if(nums[i] != nums[i-1]){
                nums[s] = nums[i];
                s++;
            }
        }
        return s;
    }

    /**
     * 27. Remove Element
     * 去除数组中等于目标值得元素，并返回新数组的长度
     * @param nums
     * @param val
     * @return
     */
    public int removeElement(int[] nums, int val) {
        int len = nums.length;
        int s = 0;
        for(int i = 0;i < len; i++){
            if(nums[i] != val){
                nums[s] = nums[i];
                s++;
            }
        }
        return s;
    }

    /**
     * 41. First Missing Positive,(未理解)
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        for(int i = 0;i < nums.length; i++){
            while(nums[i]!=i+1){
                if(nums[i]<=0 || nums[i] >= nums.length || nums[i] == nums[nums[i]-1])
                    break;
                int temp = nums[i];
                nums[i] = nums[nums[i]-1];
                nums[temp-1] = temp;
            }

        }
        for(int i = 0;i< nums.length; i++){
            if(nums[i]!=i+1)
                return i+1;
        }
        return nums.length+1;
    }

    /**
     * 88. Merge Sorted Array
     * 合并两个有序数组到nums1中，数组nums1大小为m+n,m和n分别是两数组初始化长度
     * @param nums1
     * @param m
     * @param nums2
     * @param n
     */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int k = m + n - 1;
        int i = m - 1;
        int j = n - 1;

        while (k >= 0) {
            if (i < 0) {  //数组nums1先遍历完
                nums1[k--] = nums2[j--];
                continue;
            }
            if (j < 0) {  //数组nums2遍历完,说明数组nums2中数据全部放入到nums1中了，结束循环
                break;
            }
            if (nums1[i] > nums2[j]) {
                nums1[k--] = nums1[i--];
            } else {
                nums1[k--] = nums2[j--];
            }
        }
    }

    /**
     * 43. Multiply Strings ，解法不符合题意。
     * 本方法输入为两个二进制表示的字符串，求两数之积，并返回二进制字符串。
     * @param num1
     * @param num2
     * @return
     */
    public String multiply(String num1, String num2) {
        char[] s1 = num1.toCharArray();
        char[] s2 = num2.toCharArray();
        int n1 = s1.length;
        int n2 = s2.length;
        int sum1 = 0,sum2 = 0;
        for(int i=0;i<n1;i++){
            sum1 += (s1[i]-'0') * Math.pow(2,(n1-i-1));
        }
        for(int i=0;i<n2;i++){
            sum2 += (s2[i]-'0') * Math.pow(2,(n2-i-1));
        }
        int p = sum1*sum2;
        int sh = p/2, yu = p%2;   //10进制转2进制
        StringBuilder strb = new StringBuilder();
        while(sh >= 2){
            strb.append(yu);
            yu = sh%2;
            sh = sh/2;
        }
        strb.append(yu);
        strb.append(sh);
        return strb.reverse().toString();
    }

    /**
     * 在二维数组中查找目标值，数组每一行从左到右递增，每一列从上到下递增
     * @param arr
     * @param target
     * @return
     */
    public boolean search(int[][] arr, int target){
        if (arr == null){
            return false;
        }
        int c = arr[0].length-1,r = 0;
        while (c >= 0 && r < arr.length ){
            if (arr[r][c] == target){
                return true;
            }else if (arr[r][c] > target){
                c--;
            }else {
                r++;
            }
        }
        return false;
    }

    /**
     * 判断一个数是否是素数
     * 思路：一个数只有1和其本身两个约数，则是素数。用数x除以2到x/2，若没有余数表明为素数
     * @param x
     * @return
     */
    public boolean isPrimeNumber(int x){
        if (x <= 1){
            return  false;
        }
        if (x == 2) {
            return true;
        }
        for (int i = 2 ; i<= x/2;i++){
            if (x % i == 0){
                return false;
            }
        }
        return true;
    }

    /**
     * 题目：将一个正整数分解质因数。例如：输入90,打印出90=2*3*3*5
     *(1)如果这个质数恰等于n，则说明分解质因数的过程已经结束，打印出即可。
     * (2)如果n>k，但n能被k整除，则应打印出k的值，并用n除以k的商,作为新的正整数你n,重复执行第一步。
     * (3)如果n不能被k整除，则用k+1作为k的值,重复执行第一步。
     * @param x
     * @return
     */
    public String dividePrimeFactor(int x){
        StringBuilder stringBuilder = new StringBuilder(x + "=");
        int i = 2; // 最小素数
        while (i <= x) {
            // 若x 能整除 i ，则i 是x 的一个因数
            if (x % i == 0) {
                stringBuilder.append(i + "*");
                x = x / i;
                i = 2;
            } else {
                i++;
            }
        }
        String s = stringBuilder.toString().substring(0,stringBuilder.length()-1);//不能分解时，输出x本身
        return s;
    }


    /**
     * 求两个数的最大公约数
     * @param m
     * @param n
     * @return
     */
    public int maxCommonDivisor(int m ,int n){
        if (m<n){
            int temp = m;
            m = n;
            n = temp;
        }
        while (m % n != 0){
            int temp = m % n;
            m = n;
            n = temp;
        }
        return  n;
    }

    /**
     * 求两个数的最小公倍数
     * @param m
     * @param n
     * @return
     */
    public int minCommonMultiple(int m, int n){
        return m * n /maxCommonDivisor(m,n);
    }

    /**
     * 古典问题：有一对兔子，从出生后第3个月起每个月都生一对兔子，小兔子长到第三个月后每个月又生一
     * 对兔子，假如兔子都不死，问每个月的兔子总数为多少对？
     * 兔子的规律为数列1,1,2,3,5,8,13,21...
     * @param x 月数 从1计数
     * @return
     */
    public int rabbitNum(int x){
        if (x == 1 || x==2){
            return 1;
        }else {
            return rabbitNum(x-1) + rabbitNum(x-2);
        }
    }

    /**
     * 实现函数，将输入字符串中的空格替换为%20
     * @param s
     * @return
     */
    public String replaceBlank(String s){
        if (s == null){
            return null;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0;i < s.length(); i++){
            if (s.charAt(i) == ' '){
                sb.append("%20");
            }else {
                sb.append(String.valueOf(s.charAt(i)));
            }
        }
        return sb.toString();
    }

    /**
     * 求 s=a+aa+aaa+aaaa+aa...a 的值,a取值为1~9
     * @param a
     * @param n 表示最后一个数字中a的个数
     * @return
     */
    public int sum(int a ,int n){
        int temp = 0;
        int sum = 0;
        for (int i = 1; i<=n;i++){
            temp = temp * 10 + a;
            sum = sum + temp;
        }
        return sum;
    }

    /**
     * 判断一个数是否是完数，即一个数的不包含其本身的所有约数之和等于其本身
     * 如：6 =1 + 2 +3
     * @param x
     * @return
     */
    public boolean isPerfectNum(int x){
        int temp = 0;
        for (int i = 1; i <= x/2 ;i++){
            if (x % i == 0){
                temp = temp + i;
            }
        }
        if (temp == x){
            return true;
        }else {
            return false;
        }
    }

    /**
     * 有 1、 2、 3、 4 个数字，能组成多少个互不相同且无重复数字的三位数？都是多少？
     * @param arr
     * @return
     */
    public int sum(int[] arr){
        int count =0;
        for (int i = 1;i<= 4 ; i++){
            for (int j = 1;j<= 4;j++){
                for (int k = 1;k<=4 ;k++) {
                    if (i != j && i!= k && j!= k){
                          count++;
                        System.out.println(100*i+10*j+k);
                    }
                }
            }
        }
        return count;
    }

    /**
     * 求解一个数x，使x+100是一个完全平方数，x+168也是一个完全平方数
     * @return
     */
    public int answer(){
        int a,b;
        for (int x = 1; x < 1000;x++){
             a = (int)Math.sqrt(x+100);
             b = (int)Math.sqrt(x+168);
             if (a * a == (x+100) && b * b == (x+168) ){
                 return x;
             }
        }
        return -1;
    }

    /**
     * 判断闰年的标准:1、能整除4且不能整除100 2、能整除400
     * @param year
     * @return
     */
    public boolean isLeapYear(int year){
        if ((year % 4 == 0 && year % 100 !=0) || year % 400 == 0){
            return true;
        }
        return false;
    }

    /**
     * 输入年月日，判断是一年的第几天
     * @param str
     * @return
     */
    public int dayOfYear(String[] str){
        int year = Integer.parseInt(str[0]);
        int month = Integer.parseInt(str[1]);
        int day = Integer.parseInt(str[2]);
        int currentday = 0;
        switch (month){
            case 1:
                currentday = day;
                break;
            case 2:
                currentday = 31 + day;
                break;
            case 3:
                currentday = 59 + day;
                break;
            case 4:
                currentday = 90 + day;
                break;
            case 5:
                currentday = 120 + day;
                break;
            case 6:
                currentday = 151 + day;
                break;
            case 7:
                currentday = 181 + day;
                break;
            case 8:
                currentday = 212 + day;
                break;
            case 9:
                currentday = 243 + day;
                break;
            case 10:
                currentday = 273 + day;
                break;
            case 11:
                currentday = 304 + day;
                break;
            case 12:
                currentday = 334 + day;
                break;
        }
        if (isLeapYear(year)){
            currentday++;
        }
        return currentday;
    }

    /**
     * 输入三个整数，排序后按照从小到大输出
     * @param a
     * @return
     */
    public int[] threeSort(int[] a){
        int x = a[0];
        int y = a[1];
        int z = a[2];
        if (x > y ){
            int temp = y;
            y = x;
            x = temp;
        }
        if (x > z){
            int temp = z;
            z = x;
            x = temp;
        }
        if (y > z){
            int temp = z;
            z = y;
            y = temp;
        }
        return new int[]{x,y,z};
    }

    /**
     * 剑指offer
     * 题目：一个二维数组，每一行从左到右递增，每一列从上到下递增．输
     * 入一个二维数组和一个整数，判断数组中是否含有整数
     * 解题思路： 从数组右上角元素开始查找
     * @param array
     * @param target
     * @return
     */

    public boolean find(int[][] array, int target){
        if(array == null) {
            return false;
        }
        int c = array[0].length-1,r = 0;
        while(r<array.length && c>=0){
            if(array[r][c] == target){
                return true;
            }else if(array[r][c] < target){
                r++;
            }else{
                c--;
            }
        }
        return false;
    }

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> p = new ArrayList<Integer>();
        ArrayList p2 = p;
        ListNode temp = listNode;
        while(temp != null){
            p.add(temp.val);
            temp = temp.next;
        }
        for(int i = 0; i < p.size(); i++){
            p2.add(p.get(i));
        }
        return p2;
    }
    /**
     * 冒泡排序法：
     * 思路:两两进行比较，数值大的放在后面，执行一次循环后，最大数值位于最后一位。
     */
    public void bubbleSort(int[] arr){
        int k = arr.length;
        while(k > 1) {
            for (int i = 0; i <= k-2; i++) {
                if (arr[i] > arr[i + 1]) {
                    int temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
            }
            k--;
        }
    }

    /**
     * 插入排序法：
     * 插入排序 每一步都将一个待排数据按其大小插入到已经排序的数据中的适当位置，直到全部插入完毕
     * @param arr
     */
    public void insertSort(int[] arr){
        int len = arr.length;
        for (int lastSortedIndex = 0; lastSortedIndex < len-1; lastSortedIndex++ ) {
            int extractElement = arr[lastSortedIndex+1];
            for (int j = lastSortedIndex; j>=0 ; j--){
                if (arr[j] > extractElement){
                    arr[j+1] = arr[j];
                }else {
                    arr[j+1] = extractElement;
                    break;
                }
            }
        }
    }

    /**
     * 选择排序法：每次循环比较得到未排序序列的最小值，将最小值放在对应的位置上
     * @param arr
     */
    public void selectSort(int[] arr){
        int len = arr.length;
        for (int i = 0 ; i < len-1 ; i++) {
            int currentMin = arr[i];
            int swapPosition = i;  //记录最小值位置
            for (int j = i + 1; j < len; j++) {
                if (arr[j] < currentMin) {
                    currentMin = arr[j];
                    swapPosition = j;
                }
            }
            if (swapPosition != i) {   //若当前最小值发生变化，交换元素位置
                int temp = arr[swapPosition];
                arr[swapPosition] = arr[i];
                arr[i] = temp;
            }
        }
    }

    /**
     * 合并排序：采用分治思想，将两个（或两个以上）有序表合并成一个新的有序表。
     *  即把待排序序列分为若干个子序列，每个子序列是有序的。然后再把有序子序列合并为整体有序序列。
     * @param arr
     * @param low 数组开始排序的索引
     * @param high 数组最后一个未排序的索引
     */
    public void mergeSort(int[] arr, int low, int high) {
        int mid = (low + high) / 2;
        if (low < high) {
            mergeSort(arr, low, mid); //左边
            mergeSort(arr, mid + 1, high);//右边
            mergeArr(arr, low, mid, high); //左右合并
        }
    }

    public void mergeArr(int[] arr,int low, int mid, int high){
        int[] temp = new int[high - low +1];
        int i = low;
        int j = mid  + 1;
        int k = 0;
        while (i <= mid && j <= high ){
            if (arr[i] < arr[j]){
                temp[k++] = arr[i++];
            }else {
                temp[k++] = arr[j++];
            }
        }
        while (i <= mid){
            temp[k++] = arr[i++];
        }
        while (j <= high){
            temp[k++] = arr[j++];
        }


        for (int s = 0; s< temp.length ;s++ ){
            arr[s + low] = temp[s];
        }
    }

    /**
     * 希尔排序：
     * @param arr
     */
    public void shellSort(int[] arr){

    }

    /**
     * 快速排序
     * @param arr
     */
    public void quickSort(int[] arr){

    }

    /**
     * 计数排序: 在新数组上记录原数组对应索引值的个数
     * 计数数组的长度取决于待排序数组中数据的范围（等于待排序数组的最大值与最小值的差加上1）
     * 适用情况：待排序数组范围在一个小区间，否则要为计数数组分配很多内存
     * @param arr
     */
    public void countingSort(int[] arr){
        int max, min;
        max = min = arr[0];
        for(int i = 1;i<arr.length;i++){
            if (arr[i] > max){
                max = arr[i];
            }
            if (arr[i] < min){
                min = arr[i];
            }
        }
        int[] newArr = new int[max - min +1]; //计数数组
        for(int i = 0;i<arr.length;i++){
            newArr[arr[i] - min] ++;
        }
        int j = 0;
        for(int i = 0;i< newArr.length;i++){
            while(newArr[i] > 0){
                arr[j]= i + min;
                j++;
                newArr[i]--;
            }
        }

    }

    /**
     * 基数排序
     * @param arr
     */
    public void radixSort(int[] arr){

    }

    /**
     * 堆排序
     * @param arr
     */
    public void heapSort(int[] arr){

    }

}

