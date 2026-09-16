package com.federated.fl_platform_api.dto;

import java.util.List;

/**
 * Server-side pagination envelope used by the admin directories and the
 * audit-event explorer: {@code {items, page, size, total}}. {@code page} and
 * {@code size} echo the request; {@code total} is the number of matches across
 * all pages (before slicing), so the client can render page controls.
 */
public class PagedResponseDto<T> {

    private List<T> items;
    private int page;
    private int size;
    private long total;

    public PagedResponseDto() { }

    public PagedResponseDto(List<T> items, int page, int size, long total) {
        this.items = items;
        this.page = page;
        this.size = size;
        this.total = total;
    }

    public List<T> getItems() { return items; }
    public void setItems(List<T> items) { this.items = items; }
    public int getPage() { return page; }
    public void setPage(int page) { this.page = page; }
    public int getSize() { return size; }
    public void setSize(int size) { this.size = size; }
    public long getTotal() { return total; }
    public void setTotal(long total) { this.total = total; }
}
