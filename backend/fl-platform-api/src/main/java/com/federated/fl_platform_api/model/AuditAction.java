package com.federated.fl_platform_api.model;

public enum AuditAction {
    // Identity — foundations instruments only the six tagged with [F]
    USER_REGISTERED,            // [F]
    USER_LOGIN_SUCCEEDED,       // [F]
    USER_LOGIN_FAILED,          // [F]
    USER_LOGGED_OUT,            // [F]
    USER_EMAIL_VERIFIED,
    USER_SUSPENDED,
    USER_REACTIVATED,
    USER_DELETED,
    USER_PLATFORM_ROLE_CHANGED,
    USER_PROFILE_UPDATED,
    USER_PASSWORD_CHANGED,
    // Orgs — enum reserved; instrumentation in later sub-specs
    ORG_CREATED,
    ORG_MEMBER_INVITED,
    ORG_MEMBER_JOINED,
    ORG_MEMBER_REMOVED,
    ORG_MEMBER_ROLE_CHANGED,
    ORG_OWNERSHIP_TRANSFERRED,
    // System — instrumented in foundations
    BOOTSTRAP_ADMIN_CREATED,    // [F]
    BOOTSTRAP_ORG_CREATED       // [F]
}
