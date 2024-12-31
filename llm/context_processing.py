def process_context(gps_instruction, objects, user_state):
    object_data = [
        f"{obj['object']} at {obj['distance']} on your {obj['position']}"
        for obj in objects
    ]

    prompt = f"""
    GPS: {gps_instruction}
    Objects: {', '.join(object_data)}
    User State: {user_state}
    
    Generate helpful navigation advice.
    """
    return prompt
