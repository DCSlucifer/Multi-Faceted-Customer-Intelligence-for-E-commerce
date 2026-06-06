# Architecture Diagram

Editable FigJam diagram: [Data Mining Pipeline Architecture](https://www.figma.com/board/OZvN3QYqAACLRatPzaeL2W?utm_source=other&utm_content=edit_in_figjam&oai_id=&request_id=371248dc-310e-448f-9fad-ceb4f1490599&architecture=true)

![System architecture](../reports/figures/00_architecture.png)

*Static render at `reports/figures/00_architecture.png`, used by the slide deck and
the IEEE report (which cannot display Mermaid). It is rendered directly from the
Mermaid source below via mermaid-cli — requires Node.js:*

```bash
npx -y @mermaid-js/mermaid-cli -i docs/architecture.md -o reports/figures/00_architecture.png -b white -w 2600
```

*The editable sources are the FigJam board above and the Mermaid flowchart below.*

This diagram documents the repository-level architecture for the data mining
project. It focuses on the executable pipeline, generated artifacts, report
rendering, and validation flow. Diagram labels are intentionally written in
English so the FigJam board is presentation-ready.

Update this Mermaid source first when the architecture changes, then regenerate
the FigJam diagram from the same source.

```mermaid
flowchart LR
    subgraph client ["Users and Automation"]
        researcher[Researcher]
        ci[GitHub Actions CI]
    end
    subgraph gateway ["Entrypoints"]
        pipelineCli[Pipeline CLI]
        packageBuilder[Submission Builder]
        testRunner[Test Runner]
    end
    subgraph service ["Pipeline Services"]
        preprocessing[Preprocessing and Feature Engineering]
        clustering[Customer Segmentation]
        churnModeling[Churn Modeling]
        marketBasket[Market Basket Mining]
        deepLearning[Deep Learning Extension]
        synthesis[Cross-Task Synthesis]
        renderers[Report and Slide Rendering]
        validator[Artifact Validation]
    end
    subgraph datastore ["Project Artifacts"]
        rawExcel[Raw Online Retail II Excel]
        processedData[Processed Data Store]
        models[Trained Model Store]
        reportMetrics[Metrics and Figures]
        generatedDocs[Generated README, Report, Slides]
        manifest[Manifest and Submission Package]
    end
    subgraph external ["External Systems"]
        uci[UCI Dataset Repository]
    end

    researcher -->|"Runs stages"| pipelineCli
    researcher -->|"Builds package"| packageBuilder
    ci -->|"Push or PR"| testRunner
    pipelineCli -->|"Starts pipeline"| preprocessing
    packageBuilder -->|"Full rebuild"| preprocessing
    packageBuilder -->|"Render deliverables"| renderers
    packageBuilder -->|"Final validation"| validator
    testRunner -->|"Compile and pytest"| validator
    preprocessing -.->|"Preprocessing: Dataset source"| uci
    preprocessing -->|"Reads raw Excel"| rawExcel
    preprocessing -->|"Writes clean features"| processedData
    preprocessing -->|"Feeds RFM"| clustering
    preprocessing -->|"Feeds supervised frame"| churnModeling
    preprocessing -->|"Feeds transactions"| marketBasket
    preprocessing -->|"Feeds numeric features"| deepLearning
    clustering -->|"Feeds segments"| synthesis
    clustering -->|"Writes profiles"| reportMetrics
    churnModeling -->|"Writes models"| models
    churnModeling -->|"Writes metrics"| reportMetrics
    marketBasket -->|"Writes rules"| reportMetrics
    deepLearning -->|"Writes neural artifacts"| models
    synthesis -->|"Writes combined insights"| reportMetrics
    renderers -->|"Reads snapshot"| reportMetrics
    renderers -->|"Writes deliverables"| generatedDocs
    validator -->|"Writes package status"| manifest
```
