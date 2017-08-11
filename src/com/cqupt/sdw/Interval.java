package com.cqupt.sdw;

/**
 * Created by STONE on 2017/7/25.
 */
public class Interval {

    int start;
    int end;
     Interval() { start = 0; end = 0; }
     Interval(int s, int e) { start = s; end = e; }

    @Override
    public String toString() {
        return "[" +
                 + start +
                "," + end +
                ']';
    }
}
