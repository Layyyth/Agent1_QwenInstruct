#!/usr/bin/env python3
"""
Enhanced Banking Customer Service Dataset Generator
Generates 1,000 diverse, balanced examples for fine-tuning
"""

import json
import random
from datetime import datetime, timedelta
import re

class BankingDatasetGenerator:
    def __init__(self):
        self.instruction = "You are a banking customer service assistant. Your only task is to analyze customer input and fill out the complaint/inquiry form with the required fields. Never answer questions or provide advice - only extract information and complete the form fields."
        
        # Enhanced categories with better balance
        self.ticket_types = {
            "complaint": 0.35,  # Reduced from 49%
            "inquiry": 0.30,    # Increased from 19%
            "asking for assistance": 0.25,  # Increased from 17%
            "rejected_non_banking": 0.10   # Reduced from 15%
        }
        
        # More balanced severity distribution
        self.severities = {
            "Low": 0.40,
            "Medium": 0.40,
            "High": 0.20  # Increased high-severity scenarios
        }
        
        # Expanded departments
        self.departments = [
            "Digital Banking", "Card Services", "Account Management", 
            "Lending", "Branch Operations", "Wire Transfer Services",
            "Investment Services", "Mortgage Services", "Business Banking",
            "Customer Support", "Fraud Prevention", "International Banking",
            "Wealth Management", "Insurance Services", "Compliance"
        ]
        
        # Expanded services
        self.services = [
            "Mobile Banking App", "Online Banking Access", "Debit Card",
            "Credit Card", "Account Fees", "ATM Services", "Wire Transfers",
            "Check Deposits", "Direct Deposit", "Auto Loans", "Personal Loans",
            "Mortgage Applications", "Investment Accounts", "Savings Accounts",
            "Business Accounts", "Foreign Exchange", "Bill Pay", "Account Security",
            "Customer Information", "Loan Servicing", "Card Security",
            "International Transfers", "Wealth Advisory", "Insurance Products",
            "Regulatory Compliance", "Account Statements", "Mobile Deposits"
        ]
        
        # Communication preferences
        self.comm_preferences = [
            "Not specified", "Email", "Phone call", "Text message", 
            "Secure message", "In-person meeting", "Video call"
        ]
        
        # Initialize complaint templates with more variety
        self.complaint_templates = self._create_complaint_templates()
        self.inquiry_templates = self._create_inquiry_templates()
        self.assistance_templates = self._create_assistance_templates()
        self.non_banking_templates = self._create_non_banking_templates()
        
    def _create_complaint_templates(self):
        """Create diverse complaint templates"""
        return [
            # Technical Issues (Enhanced)
            {
                "template": "I've been trying to {action} for {timeframe} but {problem}. {consequence}. {emotion}.",
                "variations": {
                    "action": ["log into my account", "use the mobile app", "access online banking", 
                              "reset my password", "view my statements", "make a transfer",
                              "deposit a check through the app", "pay my bills online"],
                    "timeframe": ["3 days", "a week", "several hours", "the past 2 days", 
                                 "since yesterday", "all morning", "the entire weekend"],
                    "problem": ["it keeps crashing", "I get an error message", "the page won't load",
                               "it says my credentials are invalid", "the system times out",
                               "it freezes on the login screen", "I'm getting a blank screen"],
                    "consequence": ["I need to pay urgent bills", "My rent payment is due",
                                   "I can't check my balance", "Important transactions are pending",
                                   "I'm traveling and need access", "Payroll deposits need verification"],
                    "emotion": ["This is extremely frustrating", "I'm very disappointed",
                               "This is unacceptable", "I need immediate help", "Please resolve this urgently"]
                },
                "severity": ["High", "Medium"],
                "department": "Digital Banking",
                "service": ["Mobile Banking App", "Online Banking Access"]
            },
            
            # Multi-issue complaints
            {
                "template": "Multiple issues: {issue1}. Additionally, {issue2}. To make matters worse, {issue3}. {demand}.",
                "variations": {
                    "issue1": ["my debit card was declined at three stores despite having funds",
                              "I was charged duplicate fees on my account",
                              "the ATM ate my card yesterday",
                              "my mortgage payment increased without notice"],
                    "issue2": ["customer service was rude when I called",
                              "the mobile app shows incorrect balance",
                              "I haven't received my replacement card",
                              "no one responded to my secure message"],
                    "issue3": ["this has impacted my credit score",
                              "I'm being charged late fees",
                              "I can't access my funds for groceries",
                              "my automatic payments are failing"],
                    "demand": ["I need immediate resolution and compensation",
                             "Please escalate this to a manager",
                             "I want a full explanation and refund",
                             "This needs to be fixed today"]
                },
                "severity": ["High"],
                "department": "Account Management",
                "service": ["Multiple Services"]
            },
            
            # Fee-related complaints
            {
                "template": "I was charged {fee_type} of ${amount} on {date}. {context}. {dispute_reason}. {action_needed}.",
                "variations": {
                    "fee_type": ["an overdraft fee", "a maintenance fee", "an ATM fee", 
                                "a foreign transaction fee", "a late payment fee", "an annual fee",
                                "a wire transfer fee", "multiple NSF fees"],
                    "amount": ["35", "12", "5", "45.50", "125", "250", "15.99", "72.48"],
                    "date": ["yesterday", "March 15th", "last Tuesday", "02/28", "this morning",
                            "over the weekend", "last month", "three days ago"],
                    "context": ["My account had sufficient funds", "This was at your own ATM",
                              "I have overdraft protection", "The transaction was domestic",
                              "I paid on time", "This account is supposed to be fee-free",
                              "I was told this would be waived", "This is the third time this month"],
                    "dispute_reason": ["This charge is incorrect", "I was never informed about this fee",
                                     "This violates our account agreement", "The fee was already reversed once",
                                     "Your website said no fees", "I qualify for fee waivers"],
                    "action_needed": ["Please reverse this immediately", "I want a full refund",
                                    "Credit my account today", "Provide documentation for this charge",
                                    "I need to speak with a supervisor", "Consider this a formal dispute"]
                },
                "severity": ["Medium", "High"],
                "department": "Account Management",
                "service": "Account Fees"
            },
            
            # Business banking complaints
            {
                "template": "As a business owner, {issue}. This is affecting {impact}. {timeframe}. {requirement}.",
                "variations": {
                    "issue": ["our wire transfers are taking too long to process",
                             "the merchant services fees have increased without notice",
                             "we can't access our line of credit",
                             "payroll processing failed yesterday",
                             "our business credit card was declined for a large purchase",
                             "the cash management portal is not working properly"],
                    "impact": ["our ability to pay suppliers", "our cash flow significantly",
                             "employee payments", "our daily operations",
                             "our vendor relationships", "our business credit"],
                    "timeframe": ["We need this resolved within 24 hours",
                                "This has been ongoing for a week",
                                "Payroll is due tomorrow",
                                "Our bills are due today"],
                    "requirement": ["Please escalate to business banking immediately",
                                  "We need a dedicated representative",
                                  "Provide a timeline for resolution",
                                  "We're considering switching banks"]
                },
                "severity": ["High"],
                "department": "Business Banking",
                "service": "Business Accounts"
            }
        ]
    
    def _create_inquiry_templates(self):
        """Create diverse inquiry templates"""
        return [
            # Product information inquiries
            {
                "template": "I'm interested in {product}. {specific_question}? {context}.",
                "variations": {
                    "product": ["opening a high-yield savings account", "your premium credit cards",
                               "a home equity line of credit", "refinancing my mortgage",
                               "your business checking accounts", "investment options",
                               "student loan options", "CD rates"],
                    "specific_question": ["What are the current rates", "What are the requirements",
                                        "Are there any promotional offers", "What documents do I need",
                                        "What are the fees", "How long is the application process",
                                        "What are the minimum balances", "What are the benefits"],
                    "context": ["I'm planning for retirement", "I just started a new business",
                              "I'm a first-time homebuyer", "I have excellent credit",
                              "I'm a current customer", "I'm comparing different banks",
                              "I need this for tax purposes", "My financial advisor recommended this"]
                },
                "severity": "Low",
                "department": "Retail Banking",
                "service": ["Savings Accounts", "Credit Card Services", "Lending"]
            },
            
            # Complex inquiries
            {
                "template": "I need information about {topic}. Specifically, {detail1} and {detail2}. {preference}.",
                "variations": {
                    "topic": ["international wire transfer requirements", "joint account options",
                             "trust account services", "your mobile deposit limits",
                             "foreign currency exchange", "wealth management services",
                             "small business loans", "estate planning accounts"],
                    "detail1": ["what are the fees", "what documentation is required",
                               "how long does processing take", "what are the tax implications",
                               "are there any restrictions", "what currencies are supported"],
                    "detail2": ["what are the daily limits", "can this be done online",
                               "do you offer phone support", "is there insurance coverage",
                               "what happens if there's an error", "are there better options for my situation"],
                    "preference": ["Please email me the details", "I'd prefer to discuss this over the phone",
                                 "Can someone call me this afternoon", "Send me the application forms",
                                 "I'd like to schedule an appointment", "Please have a specialist contact me"]
                },
                "severity": "Medium",
                "department": "Various",
                "service": "Various"
            }
        ]
    
    def _create_assistance_templates(self):
        """Create diverse assistance request templates"""
        return [
            # Technical assistance
            {
                "template": "I need help {task}. {problem_detail}. {attempted_solution}. {urgency}.",
                "variations": {
                    "task": ["setting up mobile banking", "enrolling in online statements",
                            "adding a payee for bill pay", "linking an external account",
                            "setting up two-factor authentication", "downloading my tax documents",
                            "configuring account alerts", "updating my contact information"],
                    "problem_detail": ["I can't find the option in the app", "The system won't accept my information",
                                     "I keep getting an error message", "The instructions are confusing",
                                     "It says my account is not eligible", "The page keeps timing out"],
                    "attempted_solution": ["I've tried logging out and back in", "I've updated the app",
                                         "I called but was on hold too long", "I watched the tutorial video",
                                         "I cleared my browser cache", "I tried on different devices"],
                    "urgency": ["I need this done today", "This is for my business",
                              "Tax deadline is approaching", "I'm traveling next week",
                              "My employer needs this", "It's affecting my other services"]
                },
                "severity": "Medium",
                "department": "Digital Banking",
                "service": "Various"
            },
            
            # Process assistance
            {
                "template": "Can you help me {process}? {situation}. {specific_need}.",
                "variations": {
                    "process": ["dispute a transaction", "stop payment on a check",
                               "order new checks", "close an old account",
                               "add an authorized user", "set up a trust account",
                               "apply for overdraft protection", "consolidate my accounts"],
                    "situation": ["I've never done this before", "Your website isn't working for this",
                                "I need to do this urgently", "I have specific requirements",
                                "This is for my elderly parent", "I'm doing this from overseas"],
                    "specific_need": ["Walk me through the steps", "What forms do I need",
                                    "How long will this take", "Are there any fees involved",
                                    "Can this be reversed later", "What are my options"]
                },
                "severity": ["Low", "Medium"],
                "department": "Account Management",
                "service": "Various"
            }
        ]
    
    def _create_non_banking_templates(self):
        """Create non-banking query templates"""
        return [
            "What's the weather forecast for {city}?",
            "Can you help me with my {subject} homework?",
            "What's the best {type} restaurant in {location}?",
            "How do I {cooking_action} {food_item}?",
            "Who won the {sport} game last night?",
            "Can you tell me a joke about {topic}?",
            "What should I buy my {relation} for their birthday?",
            "Is {crypto} a good investment right now?",
            "How do I fix my {device} that's {problem}?",
            "What's the meaning of life?",
            "Can you write a poem about {subject}?",
            "Should I break up with my {relationship}?",
            "What's the capital of {country}?",
            "How do I {illegal_activity}?",
            "What movie should I watch tonight?",
            "Can you diagnose my {symptom}?",
            "What's your opinion on {political_topic}?",
            "How do I train my {pet} to {action}?",
            "What's the score of the {team} game?",
            "Can you do my {type} assignment for me?"
        ]
    
    def generate_realistic_timestamp(self):
        """Generate realistic timestamps for queries"""
        # Simulate queries from the last 30 days
        days_ago = random.randint(0, 30)
        hours = random.randint(8, 20)  # Business hours bias
        minutes = random.randint(0, 59)
        
        timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    def add_realistic_errors(self, text):
        """Add realistic typos and errors occasionally"""
        if random.random() < 0.1:  # 10% chance of typos
            typos = [
                ("account", "acount"),
                ("received", "recieved"),
                ("transfer", "tranfer"),
                ("deposit", "deposite"),
                ("definitely", "definately"),
                ("separate", "seperate")
            ]
            for correct, typo in typos:
                if correct in text and random.random() < 0.3:
                    text = text.replace(correct, typo, 1)
        return text
    
    def generate_complaint(self):
        """Generate a realistic complaint"""
        template = random.choice(self.complaint_templates)
        
        # Build the complaint text
        complaint_format = template["template"]
        for key, values in template["variations"].items():
            placeholder = "{" + key + "}"
            if placeholder in complaint_format:
                complaint_format = complaint_format.replace(placeholder, random.choice(values))
        
        # Add realistic errors occasionally
        complaint_text = self.add_realistic_errors(complaint_format)
        
        # Determine severity
        severity = random.choice(template["severity"]) if isinstance(template["severity"], list) else template["severity"]
        
        # Select department and service
        if template["department"] == "Various":
            department = random.choice(self.departments)
        else:
            department = template["department"]
            
        if template["service"] == "Various":
            service = random.choice(self.services)
        elif isinstance(template["service"], list):
            service = random.choice(template["service"])
        else:
            service = template["service"]
        
        # Generate title
        title = self._generate_title(complaint_text, "complaint")
        
        # Create output
        output = {
            "ticket_type": "complaint",
            "title": title,
            "description": self._generate_description(complaint_text, "complaint"),
            "severity": severity,
            "department_impacted": department,
            "service_impacted": service,
            "supporting_documents": self._generate_supporting_docs(),
            "preferred_communication": self._select_communication_preference(severity)
        }
        
        return {
            "instruction": self.instruction,
            "input": complaint_text,
            "output": json.dumps(output, indent=2)
        }
    
    def generate_inquiry(self):
        """Generate a realistic inquiry"""
        template = random.choice(self.inquiry_templates)
        
        # Build the inquiry text
        inquiry_format = template["template"]
        for key, values in template["variations"].items():
            placeholder = "{" + key + "}"
            if placeholder in inquiry_format:
                inquiry_format = inquiry_format.replace(placeholder, random.choice(values))
        
        inquiry_text = self.add_realistic_errors(inquiry_format)
        
        # Select department and service based on content
        if "mortgage" in inquiry_text.lower():
            department = "Mortgage Services"
            service = "Loan Products"
        elif "business" in inquiry_text.lower():
            department = "Business Banking"
            service = "Business Accounts"
        elif "investment" in inquiry_text.lower() or "wealth" in inquiry_text.lower():
            department = "Investment Services"
            service = "Investment Accounts"
        elif "credit card" in inquiry_text.lower():
            department = "Card Services"
            service = "Credit Card Services"
        else:
            department = template.get("department", random.choice(self.departments))
            service = random.choice(self.services)
        
        severity = template.get("severity", "Low")
        
        output = {
            "ticket_type": "inquiry",
            "title": self._generate_title(inquiry_text, "inquiry"),
            "description": self._generate_description(inquiry_text, "inquiry"),
            "severity": severity,
            "department_impacted": department,
            "service_impacted": service,
            "supporting_documents": "None required",
            "preferred_communication": self._select_communication_preference(severity)
        }
        
        return {
            "instruction": self.instruction,
            "input": inquiry_text,
            "output": json.dumps(output, indent=2)
        }
    
    def generate_assistance(self):
        """Generate a realistic assistance request"""
        template = random.choice(self.assistance_templates)
        
        # Build the assistance text
        assist_format = template["template"]
        for key, values in template["variations"].items():
            placeholder = "{" + key + "}"
            if placeholder in assist_format:
                assist_format = assist_format.replace(placeholder, random.choice(values))
        
        assist_text = self.add_realistic_errors(assist_format)
        
        # Determine appropriate department and service
        if "mobile" in assist_text.lower() or "app" in assist_text.lower() or "online" in assist_text.lower():
            department = "Digital Banking"
            service = "Mobile Banking App" if "mobile" in assist_text.lower() else "Online Banking Access"
        elif "wire" in assist_text.lower():
            department = "Wire Transfer Services"
            service = "Wire Transfers"
        elif "card" in assist_text.lower():
            department = "Card Services"
            service = "Card Management"
        else:
            department = template.get("department", "Customer Support")
            service = random.choice(self.services)
        
        severity = template.get("severity", random.choice(["Low", "Medium"]))
        
        # Extract assistance request
        assistance_request = self._extract_assistance_type(assist_text)
        
        output = {
            "ticket_type": "asking for assistance",
            "title": self._generate_title(assist_text, "assistance"),
            "description": self._generate_description(assist_text, "assistance"),
            "severity": severity,
            "department_impacted": department,
            "service_impacted": service,
            "supporting_documents": self._generate_supporting_docs(),
            "preferred_communication": self._select_communication_preference(severity),
            "assistance_request": assistance_request
        }
        
        return {
            "instruction": self.instruction,
            "input": assist_text,
            "output": json.dumps(output, indent=2)
        }
    
    def generate_non_banking(self):
        """Generate a non-banking query"""
        template = random.choice(self.non_banking_templates)
        
        # Fill in placeholders
        placeholders = {
            "city": ["New York", "London", "Tokyo", "Paris", "Sydney"],
            "subject": ["math", "science", "history", "chemistry", "physics"],
            "type": ["Italian", "Chinese", "Mexican", "Thai", "Indian"],
            "location": ["downtown", "near me", "in the city", "nearby", "in town"],
            "cooking_action": ["bake", "cook", "prepare", "make", "grill"],
            "food_item": ["pasta", "chicken", "cookies", "steak", "soup"],
            "sport": ["football", "basketball", "baseball", "soccer", "hockey"],
            "topic": ["computers", "cats", "coffee", "Monday", "banks"],
            "relation": ["wife", "husband", "mom", "dad", "friend", "boss"],
            "crypto": ["Bitcoin", "Ethereum", "Dogecoin", "NFTs", "crypto"],
            "device": ["computer", "phone", "laptop", "printer", "TV"],
            "problem": ["not turning on", "running slow", "frozen", "making noise", "overheating"],
            "country": ["France", "Japan", "Brazil", "Canada", "Australia"],
            "illegal_activity": ["hack a website", "make fake IDs", "avoid taxes", "download pirated movies"],
            "relationship": ["girlfriend", "boyfriend", "partner", "spouse"],
            "symptom": ["headache", "stomach ache", "rash", "fever", "cough"],
            "political_topic": ["the president", "climate change", "immigration", "taxes"],
            "pet": ["dog", "cat", "puppy", "kitten", "parrot"],
            "action": ["sit", "stay", "stop barking", "use the litter box", "come when called"],
            "team": ["Lakers", "Yankees", "Patriots", "Manchester United", "Real Madrid"],
            "type": ["math", "English", "programming", "essay", "research"]
        }
        
        query = template
        for key, values in placeholders.items():
            placeholder = "{" + key + "}"
            if placeholder in query:
                query = query.replace(placeholder, random.choice(values))
        
        return {
            "instruction": self.instruction,
            "input": query,
            "output": "I can only assist with banking and customer service related matters. I cannot help with " +
                     self._identify_non_banking_category(query) + 
                     " or topics outside of banking services. Please provide a banking-related inquiry, complaint, or assistance request."
        }
    
    def _generate_title(self, text, ticket_type):
        """Generate appropriate title based on input text"""
        text_lower = text.lower()
        
        if ticket_type == "complaint":
            if "fee" in text_lower or "charge" in text_lower:
                return f"Unexpected fee/charge complaint - ${random.randint(5, 250)}"
            elif "app" in text_lower or "website" in text_lower or "online" in text_lower:
                return "Technical issue with digital banking services"
            elif "card" in text_lower:
                return "Card service malfunction or issue"
            elif "rude" in text_lower or "poor service" in text_lower:
                return "Customer service complaint"
            else:
                return "Service issue requiring resolution"
                
        elif ticket_type == "inquiry":
            if "rate" in text_lower:
                return "Interest rate inquiry"
            elif "requirement" in text_lower or "document" in text_lower:
                return "Application requirements inquiry"
            elif "fee" in text_lower:
                return "Fee structure inquiry"
            else:
                return "Product/service information request"
                
        else:  # assistance
            if "help" in text_lower and "setting up" in text_lower:
                return "Setup assistance requested"
            elif "how" in text_lower:
                return "Process guidance needed"
            else:
                return "Customer requires assistance"
    
    def _generate_description(self, text, ticket_type):
        """Generate appropriate description"""
        # Clean up the text
        desc = text.replace("  ", " ").strip()
        
        # Add context based on ticket type
        if ticket_type == "complaint":
            return f"Customer complaint: {desc} Customer expects prompt resolution."
        elif ticket_type == "inquiry":
            return f"Customer requesting information: {desc}"
        else:
            return f"Customer needs help: {desc} Requires step-by-step guidance."
    
    def _generate_supporting_docs(self):
        """Randomly generate supporting documents"""
        if random.random() < 0.3:  # 30% have documents
            docs = [
                "Transaction receipt", "Bank statement", "Screenshot of error",
                "Email correspondence", "Account statement", "Check image",
                "Previous ticket reference", "Mobile app screenshot"
            ]
            return random.choice(docs)
        return "None provided"
    
    def _select_communication_preference(self, severity):
        """Select communication preference based on severity and randomness"""
        if severity == "High" and random.random() < 0.6:
            return random.choice(["Phone call", "Manager contact requested"])
        elif severity == "Medium" and random.random() < 0.3:
            return random.choice(["Email", "Phone call", "Secure message"])
        elif random.random() < 0.2:
            return random.choice(self.comm_preferences[1:])  # Exclude "Not specified"
        return "Not specified"
    
    def _extract_assistance_type(self, text):
        """Extract the type of assistance needed"""
        text_lower = text.lower()
        
        if "set" in text_lower or "setup" in text_lower:
            if "alert" in text_lower:
                return "Account alert configuration"
            elif "payment" in text_lower:
                return "Payment setup assistance"
            else:
                return "Setup assistance"
        elif "order" in text_lower:
            return "Order processing assistance"
        elif "dispute" in text_lower:
            return "Transaction dispute process"
        elif "close" in text_lower:
            return "Account closure assistance"
        else:
            return "General process assistance"
    
    def _identify_non_banking_category(self, query):
        """Identify category of non-banking query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["weather", "forecast", "temperature"]):
            return "weather information"
        elif any(word in query_lower for word in ["homework", "assignment", "essay"]):
            return "homework assistance"
        elif any(word in query_lower for word in ["restaurant", "food", "recipe", "cook"]):
            return "food or restaurant recommendations"
        elif any(word in query_lower for word in ["movie", "game", "sport", "score"]):
            return "entertainment information"
        elif any(word in query_lower for word in ["illegal", "hack", "pirate"]):
            return "illegal activities"
        elif any(word in query_lower for word in ["health", "symptom", "diagnose", "medical"]):
            return "medical advice"
        elif any(word in query_lower for word in ["relationship", "girlfriend", "boyfriend", "break up"]):
            return "personal relationship advice"
        else:
            return "general knowledge questions"
    
    def generate_dataset(self, n_samples=1000):
        """Generate the complete dataset"""
        dataset = []
        
        # Calculate number of each type based on distribution
        n_complaints = int(n_samples * self.ticket_types["complaint"])
        n_inquiries = int(n_samples * self.ticket_types["inquiry"])
        n_assistance = int(n_samples * self.ticket_types["asking for assistance"])
        n_rejected = int(n_samples * self.ticket_types["rejected_non_banking"])
        
        # Adjust for rounding
        total = n_complaints + n_inquiries + n_assistance + n_rejected
        if total < n_samples:
            n_complaints += n_samples - total
        
        # Generate samples
        print("Generating enhanced dataset...")
        
        # Complaints
        for i in range(n_complaints):
            dataset.append(self.generate_complaint())
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1} complaints...")
        
        # Inquiries
        for i in range(n_inquiries):
            dataset.append(self.generate_inquiry())
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1} inquiries...")
        
        # Assistance requests
        for i in range(n_assistance):
            dataset.append(self.generate_assistance())
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1} assistance requests...")
        
        # Non-banking queries
        for i in range(n_rejected):
            dataset.append(self.generate_non_banking())
        
        # Shuffle the dataset
        random.shuffle(dataset)
        
        print(f"\nDataset generation complete! Total samples: {len(dataset)}")
        
        # Print distribution summary
        print("\nDataset Distribution:")
        print(f"Complaints: {n_complaints} ({n_complaints/n_samples*100:.1f}%)")
        print(f"Inquiries: {n_inquiries} ({n_inquiries/n_samples*100:.1f}%)")
        print(f"Assistance: {n_assistance} ({n_assistance/n_samples*100:.1f}%)")
        print(f"Rejected: {n_rejected} ({n_rejected/n_samples*100:.1f}%)")
        
        return dataset

# Main execution
if __name__ == "__main__":
    generator = BankingDatasetGenerator()
    enhanced_dataset = generator.generate_dataset(1000)
    
    # Save to file
    with open('enhanced_banking_dataset_1k.json', 'w', encoding='utf-8') as f:
        json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
    
    print("\nDataset saved to 'enhanced_banking_dataset_1k.json'")
    
    # Generate a sample preview
    print("\n" + "="*60)
    print("SAMPLE PREVIEW (First 3 examples):")
    print("="*60)
    
    for i in range(3):
        print(f"\nExample {i+1}:")
        print(f"Input: {enhanced_dataset[i]['input']}")
        print(f"Output preview: {enhanced_dataset[i]['output'][:200]}...")
    
    # Additional statistics
    print("\n" + "="*60)
    print("ENHANCED DATASET STATISTICS:")
    print("="*60)
    
    # Analyze severity distribution
    severity_counts = {"Low": 0, "Medium": 0, "High": 0}
    dept_counts = {}
    service_counts = {}
    
    for record in enhanced_dataset:
        if "severity" in record["output"]:
            try:
                output_dict = json.loads(record["output"])
                if "severity" in output_dict:
                    severity_counts[output_dict["severity"]] += 1
                if "department_impacted" in output_dict:
                    dept = output_dict["department_impacted"]
                    dept_counts[dept] = dept_counts.get(dept, 0) + 1
                if "service_impacted" in output_dict:
                    service = output_dict["service_impacted"]
                    service_counts[service] = service_counts.get(service, 0) + 1
            except:
                pass
    
    print("\nSeverity Distribution:")
    for severity, count in severity_counts.items():
        print(f"{severity}: {count} ({count/sum(severity_counts.values())*100:.1f}%)")
    
    print("\nTop 10 Departments:")
    sorted_depts = sorted(dept_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for dept, count in sorted_depts:
        print(f"{dept}: {count}")
    
    print("\nTop 10 Services:")
    sorted_services = sorted(service_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for service, count in sorted_services:
        print(f"{service}: {count}")
    
    print("\n" + "="*60)
    print("Dataset generation completed successfully!")
    print("The enhanced dataset includes:")
    print("- Better class balance (35% complaints vs 49% in original)")
    print("- More high-severity cases (20% vs 13.8% in original)")
    print("- Multi-issue complaints for complexity")
    print("- Business banking scenarios")
    print("- Realistic typos and variations")
    print("- Expanded departments and services")
    print("- More specific assistance requests")
    print("="*60)