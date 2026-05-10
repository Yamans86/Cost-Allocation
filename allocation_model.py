
import pandas as pd
import pulp

# =========================
# 1. LOAD DATA
# =========================

FILE_PATH = "allocation_template_3sheets.xlsx"

projects = pd.read_excel(FILE_PATH, sheet_name="projects_budget")
employees = pd.read_excel(FILE_PATH, sheet_name="employees")
cost_centers = pd.read_excel(FILE_PATH, sheet_name="cost_centers")


# =========================
# 2. CLEAN + SPLIT FIELDS
# =========================

def split_column(df, col):
    df[col] = df[col].fillna("").apply(
        lambda x: [i.strip() for i in str(x).split(";") if i.strip()]
    )
    return df


projects = split_column(projects, "roles_or_activities")
projects = split_column(projects, "locations")
projects = split_column(projects, "cost_items")

employees = split_column(employees, "positions")
employees = split_column(employees, "locations")

cost_centers = split_column(cost_centers, "items")
cost_centers = split_column(cost_centers, "locations")


# =========================
# 3. HELPER
# =========================

def has_overlap(list1, list2):
    return len(set(list1).intersection(set(list2))) > 0


# =========================
# 4. TIME PERIODS
# =========================

periods = pd.date_range("2026-01-01", "2026-12-01", freq="MS")


# =========================
# 5. BUILD ALLOCATION UNIVERSE
# =========================

print("Building allocation universe...")

employee_allocations = []

for _, emp in employees.iterrows():
    for _, proj in projects.iterrows():

        if has_overlap(emp["positions"], proj["roles_or_activities"]) and \
           has_overlap(emp["locations"], proj["locations"]):

            for period in periods:
                employee_allocations.append({
                    "employee_id": emp["employee_id"],
                    "budget_line_id": proj["budget_line_id"],
                    "period": period,
                    "salary": emp["monthly_salary"]
                })

employee_allocations = pd.DataFrame(employee_allocations)


cost_allocations = []

for _, cc in cost_centers.iterrows():
    for _, proj in projects.iterrows():

        if has_overlap(cc["items"], proj["cost_items"]) and \
           has_overlap(cc["locations"], proj["locations"]):

            for period in periods:
                cost_allocations.append({
                    "cost_center_id": cc["cost_center_id"],
                    "budget_line_id": proj["budget_line_id"],
                    "period": period,
                    "monthly_cost": cc["monthly_cost"]
                })

cost_allocations = pd.DataFrame(cost_allocations)

print(f"Employee allocation rows: {len(employee_allocations)}")
print(f"Cost allocation rows: {len(cost_allocations)}")


# =========================
# 6. PREP BUDGET
# =========================

budget_dict = projects.set_index("budget_line_id")["total_budget"].to_dict()


# =========================
# 7. BUILD MODEL
# =========================

print("Building optimization model...")

model = pulp.LpProblem("Allocation_Model", pulp.LpMaximize)


# =========================
# 8. VARIABLES
# =========================

# Employee allocation variables
x = {}

for _, row in employee_allocations.iterrows():
    key = (row["employee_id"], row["budget_line_id"], row["period"])

    x[key] = pulp.LpVariable(
        f"x_{row['employee_id']}_{row['budget_line_id']}_{row['period']}",
        lowBound=0,
        upBound=1
    )

# Cost center variables
z = {}

for _, row in cost_allocations.iterrows():
    key = (row["cost_center_id"], row["budget_line_id"], row["period"])

    z[key] = pulp.LpVariable(
        f"z_{row['cost_center_id']}_{row['budget_line_id']}_{row['period']}",
        lowBound=0,
        upBound=1
    )


# Unused capacity variables
u_emp = {}
for emp in employees["employee_id"].unique():
    for period in periods:
        u_emp[(emp, period)] = pulp.LpVariable(
            f"unused_{emp}_{period}", lowBound=0, upBound=1
        )

u_cc = {}
for cc in cost_centers["cost_center_id"].unique():
    for period in periods:
        u_cc[(cc, period)] = pulp.LpVariable(
            f"unused_cc_{cc}_{period}", lowBound=0, upBound=1
        )


# =========================
# 9. CONSTRAINTS
# =========================

print("Adding constraints...")

# Employee capacity
for emp in employees["employee_id"].unique():
    for period in periods:

        relevant = employee_allocations[
            (employee_allocations["employee_id"] == emp) &
            (employee_allocations["period"] == period)
        ]

        model += (
            pulp.lpSum(
                x[(row["employee_id"], row["budget_line_id"], row["period"])]
                for _, row in relevant.iterrows()
            )
            + u_emp[(emp, period)]
            == 1
        )


# Cost center capacity
for cc in cost_centers["cost_center_id"].unique():
    for period in periods:

        relevant = cost_allocations[
            (cost_allocations["cost_center_id"] == cc) &
            (cost_allocations["period"] == period)
        ]

        model += (
            pulp.lpSum(
                z[(row["cost_center_id"], row["budget_line_id"], row["period"])]
                for _, row in relevant.iterrows()
            )
            + u_cc[(cc, period)]
            == 1
        )


# Budget constraints
for bl in projects["budget_line_id"].unique():

    emp_part = employee_allocations[
        employee_allocations["budget_line_id"] == bl
    ]

    cc_part = cost_allocations[
        cost_allocations["budget_line_id"] == bl
    ]

    model += (
        pulp.lpSum(
            x[(row["employee_id"], row["budget_line_id"], row["period"])] * row["salary"]
            for _, row in emp_part.iterrows()
        )
        +
        pulp.lpSum(
            z[(row["cost_center_id"], row["budget_line_id"], row["period"])] * row["monthly_cost"]
            for _, row in cc_part.iterrows()
        )
        <= budget_dict.get(bl, 0)
    )


# =========================
# 10. OBJECTIVE
# =========================

print("Setting objective...")

alpha = 1.0
beta = 2.0

model += (
    alpha * (
        pulp.lpSum(
            x[(row["employee_id"], row["budget_line_id"], row["period"])] * row["salary"]
            for _, row in employee_allocations.iterrows()
        )
        +
        pulp.lpSum(
            z[(row["cost_center_id"], row["budget_line_id"], row["period"])] * row["monthly_cost"]
            for _, row in cost_allocations.iterrows()
        )
    )
    -
    beta * (
        pulp.lpSum(u_emp.values()) +
        pulp.lpSum(u_cc.values())
    )
)


# =========================
# 11. SOLVE
# =========================

print("Solving model... (this may take time)")

model.solve(pulp.PULP_CBC_CMD(msg=1))

print("Status:", pulp.LpStatus[model.status])


# =========================
# 12. EXTRACT RESULTS
# =========================

print("Extracting results...")

emp_results = []
for key, var in x.items():
    if var.value() is not None and var.value() > 0:
        emp, bl, period = key
        emp_results.append([emp, bl, period, var.value()])

emp_results = pd.DataFrame(emp_results, columns=[
    "employee_id", "budget_line_id", "period", "allocation_pct"
])


cc_results = []
for key, var in z.items():
    if var.value() is not None and var.value() > 0:
        cc, bl, period = key
        cc_results.append([cc, bl, period, var.value()])

cc_results = pd.DataFrame(cc_results, columns=[
    "cost_center_id", "budget_line_id", "period", "allocation_pct"
])


# =========================
# 13. SAVE OUTPUT
# =========================

emp_results.to_csv("employee_allocations.csv", index=False)
cc_results.to_csv("cost_allocations.csv", index=False)

print("Done. Files saved:")
print("- employee_allocations.csv")
print("- cost_allocations.csv")