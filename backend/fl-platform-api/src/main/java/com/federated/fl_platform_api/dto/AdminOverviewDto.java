package com.federated.fl_platform_api.dto;

/**
 * Aggregate platform snapshot for the admin dashboard: how many owners exist,
 * how much is pending an admin decision, and what is currently running. Lets the
 * admin "see the whole list of who owns / trains / uses" at a glance before
 * drilling into the per-list endpoints.
 */
public class AdminOverviewDto {
    private long totalUsers;
    private long owners;             // platform_role = PROJECT_OWNER
    private long admins;             // platform_role = PLATFORM_ADMIN
    private long totalProjects;
    private long runningProjects;
    private long pendingOwnerRequests;
    private long pendingDeletionRequests;
    private long pendingAccessRequests;

    public long getTotalUsers() { return totalUsers; }
    public void setTotalUsers(long totalUsers) { this.totalUsers = totalUsers; }
    public long getOwners() { return owners; }
    public void setOwners(long owners) { this.owners = owners; }
    public long getAdmins() { return admins; }
    public void setAdmins(long admins) { this.admins = admins; }
    public long getTotalProjects() { return totalProjects; }
    public void setTotalProjects(long totalProjects) { this.totalProjects = totalProjects; }
    public long getRunningProjects() { return runningProjects; }
    public void setRunningProjects(long runningProjects) { this.runningProjects = runningProjects; }
    public long getPendingOwnerRequests() { return pendingOwnerRequests; }
    public void setPendingOwnerRequests(long pendingOwnerRequests) { this.pendingOwnerRequests = pendingOwnerRequests; }
    public long getPendingDeletionRequests() { return pendingDeletionRequests; }
    public void setPendingDeletionRequests(long pendingDeletionRequests) { this.pendingDeletionRequests = pendingDeletionRequests; }
    public long getPendingAccessRequests() { return pendingAccessRequests; }
    public void setPendingAccessRequests(long pendingAccessRequests) { this.pendingAccessRequests = pendingAccessRequests; }
}
