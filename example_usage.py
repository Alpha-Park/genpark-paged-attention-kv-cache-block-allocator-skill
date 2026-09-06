import json
from client import PagedAttentionKVCacheBlockAllocator

def main():
    allocator = PagedAttentionKVCacheBlockAllocator(block_size_tokens=16, total_physical_blocks=20)
    # Allocate 35 tokens -> needs 3 blocks (48 token capacity)
    res = allocator.allocate_sequence("req_001", prompt_token_count=35)
    print("Sequence Allocation:", json.dumps(res, indent=2))
    assert res["status"] == "ALLOCATED"
    assert res["blocks_allocated"] == 3
    
    # Append tokens until boundary exceeded
    app_res = allocator.append_token("req_001", current_token_count=48)
    print("Append Token Result:", json.dumps(app_res, indent=2))
    assert app_res["status"] == "EXPANDED"
    
    # Free
    free_res = allocator.free_sequence("req_001")
    assert free_res["status"] == "FREED"
    assert free_res["free_blocks"] == 20
    print("PagedAttention allocator verification: PASS")

if __name__ == "__main__":
    main()
