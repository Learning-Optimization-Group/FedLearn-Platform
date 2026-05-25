package com.federated.fl_platform_api.model;

public enum OrgRole {
    OWNER, ADMIN, MEMBER;

    public boolean atLeast(OrgRole minimum) {
        return this.ordinal() <= minimum.ordinal();
    }
}
