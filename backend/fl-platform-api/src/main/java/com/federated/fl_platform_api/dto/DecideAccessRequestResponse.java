package com.federated.fl_platform_api.dto;

public class DecideAccessRequestResponse {
    private AccessRequestDto request;
    private MembershipDto membership;

    public AccessRequestDto getRequest() { return request; }
    public void setRequest(AccessRequestDto request) { this.request = request; }
    public MembershipDto getMembership() { return membership; }
    public void setMembership(MembershipDto membership) { this.membership = membership; }
}
