# Entity-Relationship Diagram

## Candidate Domain Model

```mermaid
erDiagram
    CANDIDATE ||--|| PROFILE : has
    CANDIDATE ||--|{ CAREER_HISTORY : has
    CANDIDATE ||--|{ EDUCATION : has
    CANDIDATE ||--|{ SKILL : has
    CANDIDATE ||--o{ CERTIFICATION : has
    CANDIDATE ||--o{ LANGUAGE : has
    CANDIDATE ||--|| REDROB_SIGNALS : has

    CANDIDATE {
        string candidate_id PK
    }

    PROFILE {
        string anonymized_name
        string headline
        string summary
        string location
        string country
        float years_of_experience
        string current_title
        string current_company
        string current_company_size
        string current_industry
    }

    CAREER_HISTORY {
        string company
        string title
        date start_date
        date end_date
        int duration_months
        bool is_current
        string industry
        string company_size
        text description
    }

    EDUCATION {
        string institution
        string degree
        string field_of_study
        int start_year
        int end_year
        string grade
        string tier
    }

    SKILL {
        string name
        string proficiency
        int endorsements
        int duration_months
    }

    CERTIFICATION {
        string name
        string issuer
        int year
    }

    LANGUAGE {
        string language
        string proficiency
    }

    REDROB_SIGNALS {
        float profile_completeness_score
        date signup_date
        date last_active_date
        bool open_to_work_flag
        int profile_views_received_30d
        int applications_submitted_30d
        float recruiter_response_rate
        float avg_response_time_hours
        json skill_assessment_scores
        int connection_count
        int endorsements_received
        int notice_period_days
        json expected_salary_range_inr_lpa
        string preferred_work_mode
        bool willing_to_relocate
        float github_activity_score
        int search_appearance_30d
        int saved_by_recruiters_30d
        float interview_completion_rate
        float offer_acceptance_rate
        bool verified_email
        bool verified_phone
        bool linkedin_connected
    }
```

## Job Understanding Output

```mermaid
erDiagram
    JOB_PROFILE ||--|{ REQUIRED_SKILL : contains
    JOB_PROFILE ||--|{ PREFERRED_SKILL : contains
    JOB_PROFILE ||--|{ BEHAVIOR_TRAIT : requires

    JOB_PROFILE {
        string domain
        string experience
        float experience_min_years
        float experience_max_years
        text raw_text
    }
```

## Ranking Output

```mermaid
erDiagram
    RANKED_CANDIDATE ||--|| COMPONENT_SCORES : has
    RANKED_CANDIDATE ||--o{ STRENGTH : explains
    RANKED_CANDIDATE ||--o{ WEAKNESS : explains
    RANKED_CANDIDATE ||--o{ MISSING_SKILL : identifies

    RANKED_CANDIDATE {
        string candidate_id FK
        int rank
        float score
        float submission_score
        string recommendation
        text reasoning
    }

    COMPONENT_SCORES {
        float skill_match
        float project_relevance
        float experience_match
        float behavior_match
        float learning_potential
    }
```

## Missing Value Summary (100K corpus sample)

| Field | Missing Rate (approx.) |
|-------|------------------------|
| certifications | ~high (many empty arrays) |
| skill_assessment_scores | ~majority empty |
| github_activity_score | -1 when no GitHub |
| offer_acceptance_rate | -1 when no offer history |
| end_date (current role) | null for current jobs |

## Candidate vs Job Field Mapping

| Job Field | Candidate Sources |
|-----------|-------------------|
| Required skills | skills[], career_history.description, skill_assessment_scores |
| Experience band | profile.years_of_experience, career_history.duration_months |
| Domain | profile.current_industry, education.field_of_study |
| Behavioral traits | profile.summary, redrob_signals.* |
| Project relevance | career_history.description (semantic match) |
| Learning potential | github_activity_score, certifications, profile_completeness_score |
