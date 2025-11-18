import graphviz

def visualize_infra_state_machine():
    dot = graphviz.Digraph(comment='Multi-Region Application Infrastructure State Machine', graph_attr={'rankdir': 'LR'})

    # Define the primary components and their states
    components = {
        "Region A VPC": "Created",
        "Region B VPC": "Created",
        "Region A Load Balancer": "Created",
        "Region B Load Balancer": "Created",
        "Region A Web Server": "Created",
        "Region B Web Server": "Created",
        "Region A Database": "Created",
        "Region B Database": "Created",
    }

    # Add nodes for each component
    for component, state in components.items():
        dot.node(component, f"{component}\n({state})", shape='box')

    # Add conceptual initial and final states
    dot.node("Start", "Terraform Apply\n(Initial State)", shape='oval', style='filled', fillcolor='lightgreen')
    dot.node("End", "Infrastructure Deployed\n(Final State)", shape='oval', style='filled', fillcolor='lightblue')
    dot.edge("Start", "Region A VPC")
    dot.edge("Start", "Region B VPC")


    # Define dependencies and transitions (simplified)
    # VPCs must exist before other resources in that region
    dot.edge("Region A VPC", "Region A Load Balancer", label="depends on")
    dot.edge("Region A VPC", "Region A Web Server", label="depends on")
    dot.edge("Region A VPC", "Region A Database", label="depends on")

    dot.edge("Region B VPC", "Region B Load Balancer", label="depends on")
    dot.edge("Region B VPC", "Region B Web Server", label="depends on")
    dot.edge("Region B VPC", "Region B Database", label="depends on")

    # Web servers often depend on Load Balancers (to be targeted) and Databases (to connect)
    dot.edge("Region A Load Balancer", "Region A Web Server", label="targets")
    dot.edge("Region A Web Server", "Region A Database", label="connects to")

    dot.edge("Region B Load Balancer", "Region B Web Server", label="targets")
    dot.edge("Region B Web Server", "Region B Database", label="connects to")

    # Conceptual transition to final state (all resources created)
    # This is a simplification; in reality, all dependencies would need to resolve
    dot.edge("Region A Database", "End", label="regional done", arrowhead="empty")
    dot.edge("Region B Database", "End", label="regional done", arrowhead="empty")
    dot.edge("Region A Web Server", "End", label="regional done", arrowhead="empty")
    dot.edge("Region B Web Server", "End", label="regional done", arrowhead="empty")
    dot.edge("Region A Load Balancer", "End", label="regional done", arrowhead="empty")
    dot.edge("Region B Load Balancer", "End", label="regional done", arrowhead="empty")


    dot.render('multi_region_infra_state_machine', view=False, format='png')
    print("Generated 'multi_region_infra_state_machine.png'")

if __name__ == "__main__":
    visualize_infra_state_machine()
  
