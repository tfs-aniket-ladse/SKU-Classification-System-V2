# Microsoft Teams Integration Configuration
# 
# To set up Teams notifications:
# 1. Go to your Teams channel
# 2. Click on "..." (More options) next to the channel name
# 3. Select "Connectors"
# 4. Find "Incoming Webhook" and click "Configure"
# 5. Give it a name like "SKU Classification System"
# 6. Copy the webhook URL and replace the placeholder below

# Teams Webhook URL - Replace with your actual webhook URL
TEAMS_WEBHOOK_URL = "https://thermofisher.webhook.office.com/webhookb2/2a9e5d20-d044-4e5c-90dc-b647b766e24f@b67d722d-aa8a-4777-a169-ebeb7a6a3b67/IncomingWebhook/b86f8c74fdc14004b6bbb89e8d005d7f/24b25157-22c4-467c-80d7-2a7ae4c52939/V2tqepN4w8hiweI_jlpRgf7KnNY5hfE2Gm8YAk9tVF5pQ1"

# Example webhook URL format:
# TEAMS_WEBHOOK_URL = "https://outlook.office.com/webhook/your-webhook-id-here"

# To use this configuration:
# 1. Update the webhook URL above
# 2. In bulk_v1.py, replace the placeholder webhook_url with:
#    from teams_config import TEAMS_WEBHOOK_URL
#    webhook_url = TEAMS_WEBHOOK_URL

# Test message function
def send_feedback_notification(user_email, user_name, sku_input, name_input, prediction, feedback_type, correct_product_line=None, correct_business_unit=None, user_comment=None):
    """Send feedback notification to Teams"""
    import requests
    from datetime import datetime
    
    if TEAMS_WEBHOOK_URL == "YOUR_TEAMS_WEBHOOK_URL_HERE":
        return True  # Skip if not configured
    
    # Color coding for feedback
    theme_color = "28A745" if feedback_type == "like" else "DC3545"  # Green for like, Red for dislike
    feedback_emoji = "👍" if feedback_type == "like" else "👎"
    
    # Build facts list
    facts = [
        {"name": "User Email", "value": user_email},
        {"name": "User Name", "value": user_name},
        {"name": "Feedback Type", "value": f"{feedback_emoji} {feedback_type.upper()}"},
        {"name": "Input SKU Number", "value": sku_input or "N/A"},
        {"name": "Input SKU Name", "value": name_input or "N/A"},
        {"name": "Predicted Code", "value": prediction.get('product_line_code', 'N/A')},
        {"name": "Predicted CMR Line", "value": prediction.get('cmr_product_line', 'N/A')},
        {"name": "Confidence Score", "value": f"{prediction.get('combined_score', 0)}%"}
    ]
    
    # Add correction details for dislikes
    if feedback_type == "dislike":
        if correct_product_line:
            facts.append({"name": "Correct Product Line Code", "value": correct_product_line})
        if correct_business_unit:
            facts.append({"name": "Correct Business Unit", "value": correct_business_unit})
        if user_comment:
            facts.append({"name": "User Comment", "value": user_comment})
    
    facts.append({"name": "Timestamp", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    
    message = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "summary": f"User Feedback - {feedback_type.title()}",
        "themeColor": theme_color,
        "sections": [{
            "activityTitle": f"SKU Classification - User Feedback {feedback_emoji}",
            "activitySubtitle": f"User: {user_name} ({user_email})",
            "facts": facts
        }]
    }
    
    try:
        response = requests.post(TEAMS_WEBHOOK_URL, json=message)
        return response.status_code == 200
    except Exception as e:
        print(f"Teams feedback notification error: {str(e)}")
        return True  # Don't block user experience

def send_daily_summary_notification():
    """Send daily summary of user activity and feedback"""
    import requests
    from datetime import datetime
    import pandas as pd
    import os
    
    if TEAMS_WEBHOOK_URL == "YOUR_TEAMS_WEBHOOK_URL_HERE":
        return True
    
    try:
        # Load feedback data
        feedback_file = os.path.join('feedback_data', 'bulk_user_feedback.csv')
        if not os.path.exists(feedback_file):
            return True
        
        feedback_df = pd.read_csv(feedback_file)
        today = datetime.now().strftime('%Y-%m-%d')
        today_feedback = feedback_df[feedback_df['timestamp'].str.contains(today)]
        
        if today_feedback.empty:
            return True
        
        total_feedback = len(today_feedback)
        likes = len(today_feedback[today_feedback['feedback'] == 'like'])
        dislikes = len(today_feedback[today_feedback['feedback'] == 'dislike'])
        unique_users = today_feedback['user_email'].nunique()
        accuracy = (likes / total_feedback * 100) if total_feedback > 0 else 0
        
        message = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": "Daily SKU Classification Summary",
            "themeColor": "17A2B8",
            "sections": [{
                "activityTitle": "SKU Classification System - Daily Summary",
                "activitySubtitle": f"Activity Report for {today}",
                "facts": [
                    {"name": "Total Feedback", "value": str(total_feedback)},
                    {"name": "Unique Users", "value": str(unique_users)},
                    {"name": "Likes (Correct)", "value": f"{likes} ({likes/total_feedback*100:.1f}%)"},
                    {"name": "Dislikes (Incorrect)", "value": f"{dislikes} ({dislikes/total_feedback*100:.1f}%)"},
                    {"name": "System Accuracy", "value": f"{accuracy:.1f}%"},
                    {"name": "Report Generated", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                ]
            }]
        }
        
        response = requests.post(TEAMS_WEBHOOK_URL, json=message)
        return response.status_code == 200
        
    except Exception as e:
        print(f"Daily summary notification error: {str(e)}")
        return True

def send_streamlit_notification(sku_number, sku_name, tab_type):
    """Send notification for Streamlit single SKU requests"""
    import requests
    from datetime import datetime
    
    if not TEAMS_WEBHOOK_URL:
        return True
    
    try:
        message = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": "Streamlit SKU Request",
            "themeColor": "FF6B35",
            "sections": [{
                "activityTitle": f"Streamlit - {tab_type}",
                "activitySubtitle": f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "facts": [
                    {"name": "SKU Number", "value": sku_number or "N/A"},
                    {"name": "SKU Name", "value": sku_name or "N/A"},
                    {"name": "Tab", "value": tab_type}
                ]
            }]
        }
        requests.post(TEAMS_WEBHOOK_URL, json=message, timeout=5)
        return True
    except:
        return True

def test_teams_notification():
    """Test function to verify Teams integration"""
    import requests
    from datetime import datetime
    
    if TEAMS_WEBHOOK_URL == "YOUR_TEAMS_WEBHOOK_URL_HERE":
        print("Please update the TEAMS_WEBHOOK_URL in teams_config.py")
        return False
    
    test_message = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "summary": "Test Notification",
        "themeColor": "0076D7",
        "sections": [{
            "activityTitle": "SKU Classification System - Test",
            "activitySubtitle": "Testing Teams integration",
            "facts": [
                {"name": "Status", "value": "Test successful"},
                {"name": "Time", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            ]
        }]
    }
    
    try:
        response = requests.post(TEAMS_WEBHOOK_URL, json=test_message)
        if response.status_code == 200:
            print("Teams notification test successful!")
            return True
        else:
            print(f"Teams notification failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error testing Teams notification: {str(e)}")
        return False

if __name__ == "__main__":
    test_teams_notification()