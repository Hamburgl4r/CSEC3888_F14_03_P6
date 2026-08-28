# UG Capstone Project Weekly Meeting Minutes

## P6 - AI-Powered Legal Research and Document Assistance Platform

---

### Group Formal Tag: CSEC3888_F14_03_P6
### Tutor Name: Zhenlin Liu
### Client Name: Han Bocheng/Lujia Yang (Vecton AI Pty Ltd)

---

### Group Member Details

(indicate if any member is doing multiple units)

| No. |    Student Name    | Student ID | Student Unikey |            XP Roles           |
|:---:|:------------------:|:----------:|:--------------:|:-----------------------------:|
|  1  | Leo Tran           | 540815038  | ttra0630       |      Programmer + Manager     |
|  2  | William Duong Quan | 540754276  | wduo0657       |           Programmer          |
|  3  | Steve Le           | 530190891  | Dule3830       |       Programmer + Coach      |
|  4  | Kai Young Lee      | 530322894  | kyou7218       |      Programmer + Tester      |
|  5  | Johnny Wang        | 540709704  | jwan0989       | Programmer + Customer Liaison |
|  6  | Edward Chan        | 540709715  | echa0443       |      Programmer + Tracker     |
|  7  | Jett Vongxayasy    | 530322034  | jvon9617       |      Programmer + Tester      |

Attendances:  Leo, William, Steve, Kai, Johnny, Edward, Jett

Apologies: N/A

Submission Date: 15/08/2026

---

## Weekly Group Meeting Minutes

Time: 9-10:30pm

Venue: Microsoft Teams

Meeting Minute Taker: Steve

| # |                     Agenda Item                     |                                                                            Description / Comments                                                                            |                                                            Decision / Action                                                           |                   Who?                  |                      Items for escalation                     |
|:-:|:---------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------:|:-------------------------------------------------------------:|
| 1 | What has been completed?                            | Contacted the client to organise the initial meeting. Prepared a list of questions for the client. Communication channels established for the group.                         | Continue preparing for the initial client meeting.                                                                                     | Manager / Customer Liaison              | —                                                             |
| 2 | What is in progress?                                | Confirming a suitable time for the first client meeting.                                                                                                                     | Follow up with the client and confirm meeting date/time.                                                                               | Manager / Customer Liaison              | Escalate to tutor if unable to establish contact with client. |
| 3 | What is working well?                               | Communication has been established between group members through Microsoft Teams and Discord. Members are actively discussing project requirements and administrative tasks. | Continue using agreed communication channels and keep important project decisions documented.                                          | All group members                       | —                                                             |
| 4 | What needs improvement?                             | Client requirements and project scope have not yet been formally confirmed. Some administrative tasks are still incomplete.                                                  | Use the initial client meeting to clarify scope, priorities, requirements and expectations. Complete outstanding administrative tasks. | All group members                       | Any unclear or conflicting client requirements.               |
| 5 | Reminders                                           | Complete the Group Contract and Student Deed Poll. Prepare for the client meeting.                                                                                           | Ensure all required documents are completed and uploaded by the deadline.                                                              | All group members                       | Missing/incomplete Deed Polls or Group Contract.              |
| 6 | What needs to be completed before the next meeting? | Hold the initial client meeting. Complete and upload the Group Contract and Deed Polls. Begin reviewing the updated project brief and dataset requirements.                  | Record client decisions in meeting minutes and update project requirements accordingly.                                                | All group members                       | Any issues identified during the client meeting.              |
| 7 | What’s Next                                         | Confirm project scope and requirements with the client. Begin developing initial user stories, project scope, risks and technical research.                                  | Convert confirmed client requirements into initial user stories and tasks for the backlog.                                             | Customer Liaison/ Tracker / Programmers | Requirements requiring clarification from client/tutor.       |
| 8 | Initial technical research                          | Review the updated project brief, Open Australian Legal Corpus, retrieval requirements and citation requirements.                                                            | Allocate initial research tasks among members.                                                                                         | Programmers / Researchers               | Dataset access or technical constraints.                      |


**Main Discussions**
- Discussed preparations for the first client meeting.
- Reviewed the project brief and prepared questions for the client.
- Discussed group communication, roles, and outstanding admin tasks.

**Key Takeaways**
- Client requirements and scope still needed to be confirmed.
- Group Contract and Deed Polls needed to be completed.
- The team should prepare well before starting major development.

**What Happened**
- Client was contacted to organise the first meeting.
- Initial client questions were prepared.
- Teams and Discord communication were set up.
- Group members began reviewing the project requirements.

**What’s Next**
- Complete and upload the Group Contract and Deed Polls.
- Begin preparing the initial project scope and user stories after client requirements are confirmed.
- Organise a set, weekly schedule for client meetings.

---

## Weekly Client Meeting Minutes

Time: 1-1:20pm 15/08/2026

Venue: Microsoft Teams

Meeting Minute Taker: Leo Tran

Attendances: Leo Tran, Steve Le, Kai Young Lee, Edward Chan, Johnny Wang

Apologies: Jett Vongxayasy, William Duong Quan

| # |              Agenda Item              |                                                                                               Description / Comments                                                                                              |                             Decision / Action                            |           Who?          |                 Items for escalation                |
|:-:|:-------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------:|:-----------------------:|:---------------------------------------------------:|
| 1 | Introductions & project understanding | Briefly introduce the team and confirm our understanding of the project: a legal research web app using the Civil Liability Act 2002 (NSW) and NSW judgments, with reliable paragraph-level citations.            | Confirm that our understanding matches the client's expectations.        | Group / Client          | Any misunderstanding of the brief                   |
| 2 | Project scope & priorities            | Confirm what should be prioritised for the core system. Is retrieval → legislation/cases → exact paragraph citation the main initial workflow?                                                                    | Agree on the most important functionality to work on first.              | Client / Customer       | Scope changes or unclear requirements               |
| 3 | 14 September checkpoint expectations  | Clarify exactly what the client expects to see at the first demo. The brief currently requires working retrieval over the real corpus with citations resolving correctly; generated answers are not yet required. | Record clear acceptance criteria for the checkpoint.                     | Client                  | Requirements that cannot be completed by checkpoint |
| 4 | Evaluation approach                   | Ask how the evaluation/test set should be constructed and how citation accuracy, hallucination rate and retrieval performance should be measured.                                                                 | Agree on evaluation approach or identify what still needs to be decided. | Client / Group          | Evaluation details still TBD                        |
| 5 | Citation & hallucination requirements | What counts as a correct citation? How should unsupported claims be handled? Should the system completely refuse an answer where evidence is insufficient?                                                        | Record exact expected behaviour.                                         | Client                  | Ambiguous evaluation criteria                       |
| 6 | Technical constraints / model choices | Ask whether there are preferred/restricted LLMs, embedding models, rerankers, APIs, hosting services or budgets.                                                                                                  | Confirm whether the team is free to choose the technical stack.          | Client / Technical team | API costs, hosting restrictions                     |
| 7 | Dataset / corpus coordination         | Confirm expectations for filtering the Open Australian Legal Corpus and whether the three teams should compare judgment counts / version_id lists early.                                                          | Agree on how baseline dataset consistency will be checked.               | Client / Group          | Different dataset counts between teams              |
| 8 | Weekly communication                  | Confirm what the client wants before each weekly meeting: short written progress note, blockers, decisions needed, demos, etc.                                                                                    | Agree on meeting/update format.                                          | Manager / Client        | Communication/access issues                         |

**Main Discussions**
- Discussed project scope and client expectations.
- Clarified dataset, retrieval, citations, and evaluation.
- Discussed what is required for the first checkpoint.

**Key Takeaways**
- Prioritise accurate retrieval and citations.
- Preserve exact judgment paragraph numbers.
- Avoid unsupported or hallucinated legal claims.
- Focus on the required baseline dataset first.

**What Happened**
- Initial client meeting completed.
- Updated project brief reviewed.
- Client questions discussed.

**What’s Next**
- William: Download and filter the required legal corpus and report the judgment count.
- Kai: Test whether judgment paragraph numbers are extracted and preserved correctly.
- Steve: Compare 2–3 retrieval approaches and recommend one for the first prototype.
- Johnny: Create the initial user stories from the confirmed client requirements.
- Edward: Add the agreed tasks, owners and deadlines to the project tracker.
- Leo: Coordinate the above tasks and prepare the next client meeting.
- Jett: Help define tests for citation accuracy and retrieval results.

---

### Individual Progress Report

Name: Leo Tran

Student ID: 540815038

Goals: Get a feel of the group, allocate our roles, meet our client and to develop a to-do list for the week so we can get our capstone project in order.

Progress Made: The group has met each other and has allocated roles. I have taken a manager role for this week and have set up a basic google drive and documents folder. Our github repository has also been set up and methods of communication through discord too. Client has met with us and has supplied a new project scope which has greatly reduced the scope of the project and has been much more specific.

Blockers to Progress/Issues: Issues with time constraints due to clashes in everyone's schedule . Our client has also been slow to respond which has slowed down progress.

### Individual Progress Report

Name: Johnny Wang

Student ID: 540709704

Goals Establish scope, user stories, and project schedule.

Progress Made Emailed and contacted client, established team roles, created communication channels including background research on project scope.

Blockers to Progress/Issues Client replied fairly late to our emails regarding concerns for an initial meeting, due to this and while on fairly short notice to organise said meeting, our first client meeting did not have the whole group present.

### Individual Progress Report

Name: Kai Young Lee

Student ID: 530322894

Goals -  Meet with clients and finish setting up team resources.

Progress Made - Created team GitHub, contacted client about setting up the meeting.

Blockers to Progress/Issues - Coordinating the whole team to be present for the first client meeting.

### Individual Progress Report

Name: Jett Vongxayasy

Student ID: 530322034

Goals - Contact client, research the project to come up with questions for the client meeting, work on project scope.

Progress Made - Emailed the client, established group roles, came up with a list of questions to ask during the client meeting

Blockers to Progress/Issues - Client replied fairly late, so the initial meeting was delayed

### Individual Progress Report

Name: Edward Chan

Student ID: 540709715

Goals: discuss with the client about the project scope, make user stories, finish contract and submit deedpolls, met with group members

Progress Made We organised a group meeting with mostly everyone in our group but did not meet with the client, created a GitHub, discord and teams chat.

Blockers to Progress/Issues the client responded to us late into the week with his availability so our first meeting was delayed.

### Individual Progress Report

Name: William Duong Quan

Student ID: 540754276

Goals: Learning about group members, discussing required tasks, learning about project requirements and scope.

Progress Made: Successfully got to know group members, discussed required weekly tasks, created Microsoft teams community. Did not get to discuss the project scope.

Blockers to Progress/Issues: Needed to contact the client to be able to further understand the requirements, expected outcomes and scope of the project.

### Individual Progress Report

Name: Steve Le

Student ID: 530190891

Goals: Research Relevant technologies applicable to project. Learn and understand individual group member’s strengths and responsibilities.

Progress Made: Met with client and discuss deliverables and scope. Assign roles to group members up until week 5. Was not able to discuss further on technical details/limitations with client

Blockers to Progress/Issues: Limited knowledge on the domain of LLMs.

