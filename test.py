
import heapq
from typing import List, Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        
        if not lists or len(lists) == 0:
            return None

        heads = [(head.val, i, head) for i, head in enumerate(lists) if head is not None]
        if len(heads) == 0:
            return None
        heapq.heapify(heads)

        val, i, p = heapq.heappop(heads)
        head = p
        tail = p

        p = p.next
        if p:
            heapq.heappush(heads, (p.val, i, p))

        while len(heads) > 0:
            val, i, p = heapq.heappop(heads)
            tail.next = p
            tail = p

            p = p.next
            if p:
                heapq.heappush(heads, (p.val, i, p))
            

        return head

if __name__ == '__main__':
    s = Solution()
    l1 = ListNode(1, None)
    l2 = ListNode(0, None)

    res = s.mergeKLists([l1, l2])
    while res:
        print(res.val)
        res = res.next