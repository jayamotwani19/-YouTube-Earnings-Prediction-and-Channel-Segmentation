#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#


# Load libraries
library(shiny)
library(shinyWidgets)
library(shinydashboard)
library(pander)
library(ROCR)
library(ROCit)
library(factoextra)
library(NbClust)
library(fpc)
library (ggplot2)
library(ggthemes)
library(DT)
library(rpart)
library(rpart.plot)
library(dplyr)



# Load data
df<- read.csv("main_df.csv")
dt <- read.csv("scaled_df.csv") 
df_data = read.csv("train_df.csv")
dev_data = read.csv("train_dev.csv")
ig_data = read.csv("train_enc.csv")
df_test = read.csv("test_df.csv")
dev_test = read.csv("test_dev.csv")
enc_test = read.csv("test_enc.csv")
auc_df = read.csv("test_AUC_df.csv")
dev_df = read.csv("test_dev_df.csv")
dev_df = dev_df[,2:3]

# Features for classification model
fetur_list <- c("Feature Set 1", "Feature Set 2")

numvars <- c("subscribers", "video_views", "uploads", 
             "video_views_for_the_last_30_days", "subscribers_for_last_30_days",
             "Population", "Aggregated_GDP",  "channel_lifetime")

devsel <- c("predCountry", "predsubscribers", "predvideo_views", 
            "predvideo_views_for_the_last_30_days",
            "predsubscribers_for_last_30_days", "predchannel_lifetime")

igsel <- c("video_views_for_the_last_30_days", "subscribers_for_last_30_days",
           "uploads", "predCountry", "predchannel_type")

outcome <- "earnings_level"

pos = "1"

fV_all <- paste(outcome,' ~ ', paste(c(numvars), collapse=' + '), sep='')
fV_dev <- paste(outcome,' ~ ', paste(c(devsel), collapse=' + '), sep='')
fV_ig <- paste(outcome,' ~ ', paste(c(igsel), collapse=' + '), sep='')

# Function to create Decision Tree
makeTree = function(f, dat) {
  
  tree = rpart(f, data = dat)
  
  return(tree)
}

logLikelihood <- function(ytrue, ypred, epsilon=1e-6) {
  sum(ifelse(ytrue==pos, log(ypred+epsilon), log(1-ypred-epsilon)), na.rm=T)
}

performanceMeasures <- function(ytrue, ypred, 
                                model.name = "model", threshold=0.5) {
  # compute the normalised deviance
  dev.norm <- -2 * logLikelihood(ytrue, ypred)/length(ypred)
  # compute the confusion matrix
  cmat <- table(actual = ytrue, predicted = ypred >= threshold)
  accuracy <- sum(diag(cmat)) / sum(cmat)
  precision <- cmat[2, 2] / sum(cmat[, 2])
  recall <- cmat[2, 2] / sum(cmat[2, ])
  f1 <- 2 * precision * recall / (precision + recall)
  
  data.frame(model = model.name, precision = precision,recall = recall, 
             f1 = f1, dev.norm = dev.norm)
}

panderOpt <- function(){
  # setting up Pander Options
  panderOptions("plain.ascii", TRUE)
  panderOptions("keep.trailing.zeros", TRUE)
  panderOptions("table.style", "simple")
}

pretty_perf_table <- function(model, xtrain, ytrain, xtest, ytest, m = 0, threshold = 0.5, title = "Performance Table") {
  # Option setting for Pander
  panderOpt()
  perf_justify <- "lrrrr"
  
  # Create the title line
  cat(paste0("## ", title, "\n"))
  
  if (m == 1) {
    # comparing performance on training vs. test
    trainperf_df <- performanceMeasures(
      xtrain, ytrain, model.name = "training", threshold = threshold)
    testperf_df <- performanceMeasures(
      xtest, ytest, model.name = "test", threshold = threshold)
    # combine the two performance data frames using rbind()
    perftable <- rbind(trainperf_df, testperf_df)
    #pandoc.table(perftable, justify = perf_justify)
  } else {
    # call the predict() function to do the predictions
    pred_train <- predict(model, newdata = xtrain)
    pred_test <- predict(model, newdata = xtest)
    # comparing performance on training vs. test
    trainperf_df <- performanceMeasures(
      ytrain, pred_train, model.name = "training", threshold = threshold)
    testperf_df <- performanceMeasures(
      ytest, pred_test, model.name = "test", threshold = threshold)
    # combine the two performance data frames using rbind()
    perftable <- rbind(trainperf_df, testperf_df)
    #pandoc.table(perftable, justify = perf_justify)
  }
}


# User Interface
ui <- shinyUI(
  dashboardPage(
    
    dashboardHeader(title ="Youtube"),
    
    dashboardSidebar(
      sidebarMenu(
        menuItem("K-means clustering", tabName = "kmeans", icon = icon("users", lib = "font-awesome")),
        menuItem("Clustering Statistics-1", tabName = "clusteringstats_1", icon = icon("table")),
        menuItem("Clustering Statistics-2", tabName = "clusteringstats_2", icon = icon("table")),
        menuItem("Clustered Data", tabName = "finaldata", icon = icon("table")),
        menuItem("Single Variable Classification", tabName = "svc", icon = icon("cog")),
        menuItem("Multi Variable Classification", tabName = "mvc", icon = icon("usb"))
      )
    ),
    
    
    dashboardBody(
      tabItems(
        tabItem(tabName = "kmeans", h1("K-means clustering"),
                fluidRow(
                  box(plotOutput("clusterchart")),
                  box(sliderInput("clustnum", "Number of clusters", 1, 10, 6))
                )
        ),
        
        tabItem(tabName = "clusteringstats_1", h1("Clustering Statistics-1"),
                fluidRow(
                  column(6, tableOutput("table1")))
                
        ),
       
        tabItem(tabName = "clusteringstats_2", h1("Clustering Statistics-2"),
                fluidRow(
                  div(style = "overflow-x: scroll;", tableOutput("table2")))
                
        ),
        uiOutput("custom_style"),  # Apply custom CSS for the tables
        tabItem(tabName = "finaldata", h1("Clustered Data"),
                fluidRow(
                  div(style = "overflow-x: scroll;", tableOutput("clustdata"))
                )
        ),
        tabItem(tabName = "svc",
                h1("Single Variable Model Performanace"),
                
                fluidRow(
                  label = NULL,
                  column(6,
                         h2("AUC Score"),
                         helpText(
                           "Shown below are the AUC scores of each of the single ",
                           "variate model. Arranging the scores in descending ",
                           "order gives us the preditive power of each of the ",
                           "variables. We are looking for variables with AUC score ",
                           "above 0.5 at the least which would mean the variable ",
                           "performs better than random guessing. "
                         ),
                         
                         br(),
                         tableOutput("table4")
                  ),
                  column(6,
                         h2("Deviance Score"),
                         helpText(
                           "The deviance score of each single variable model ",
                           "provides us another window into judging the performance",
                           "of single variable models. Unlike AUC scores, the closer ",
                           "the deviance score is to zero the better and so the variable",
                           "are arranged in the ascending order of absolute value of the ",
                           "deviance score to signify decreasing performance."
                         ),
                         br(),
                         tableOutput("table5")
                  )
                ),
                
        ),
        
        tabItem(tabName = "mvc",
                h1("Decision Trees"),
                
                sidebarLayout(
                  sidebarPanel(
                    h2("Choose your feature set"),
                    br(),
                    pickerInput(
                      inputId = "fetur_set",
                      choices = fetur_list,
                      options = list(`actions-box` = TRUE)
                    ),
                    br(),
                    helpText(
                      "We have the option to select from three feature sets"
                    ),
                    helpText(
                      "Feature Set 1 - selected based on deviance score less than 0"
                      ),
                    helpText(
                      "Country , subscribers, video_views, ", 
                      "video_views_for_the_last_30_days, ",
                      "subscribers_for_last_30_days, channel_lifetime"
                    ),
                    helpText(
                      "Feature Set 2 - selected based on Information Gain score greater than 0.1"
                    ),
                    helpText(
                      "video_views_for_the_last_30_days, subscribers_for_last_30_days, ",
                      "uploads, Country, channel_type"
                    ),
                    br(),
                    actionButton(
                      inputId = "createModel",
                      label = "Create Model",
                      class = "btn-primary"
                    ),
                    br(),
                    helpText(
                      "Choose the feature set you would like to see the Decision Tree",
                      "build from the drop down list and use the Create Model button",
                      "above to create the visualization of the tree."
                    )
                  ),
                  
                  mainPanel(
                    plotOutput(outputId = "tree_plot"),
                    br(),
                    br(),
                    br(),
                    
                    h2("Performance Metrics"),
                    fluidRow(
                      div(style = "overflow-x: scroll;", tableOutput("table3")))
                  )
                )
        )
      )
    )
  )
)

# Shiny Server
server <- shinyServer(function(input,output){
  
  # Custom CSS for the tables
  output$custom_style <- renderUI({
    tags$style(HTML("
      .custom-table {
        max-width: 100%;
        overflow-x: auto;
      }
    "))
  })
  
  output$clusterchart <- renderPlot({
    fviz_cluster((eclust(dt[,1:8], "kmeans", k = input$clustnum, nstart = 25, graph = FALSE)), geom = "point", ellipse.type = "norm",
                 palette = "jco", ggtheme = theme_minimal())
    
  })
  
  cluster_results <- reactive({
    clustering_fit <- kmeans(dt[, 1:8], centers = input$clustnum, nstart = 25)
    cluster_assignments <- clustering_fit$cluster
    
    return(list(cluster_assignments, clustering_fit))
  })
  
  df_res <- df
  
  # Table1 showing how many species in each cluster
  output$table1 <- renderTable({
    df_res$cluster = cluster_results()[[1]] # add cluster assignment to df_res
    df_res %>%
      group_by(cluster, earnings_level) %>% 
      tally() # count how many Species in each cluster
  })
  
  # Table2 showing the summary statistics of each cluster
  output$table2 <- renderTable({
    df_res$cluster = cluster_results()[[1]] # add cluster assignment to df_res
    df_res %>% group_by(cluster) %>%
      summarise(n = n(),
                avg.subscribers = mean(subscribers),
                avg.subscribers_last_30_days = mean(subscribers_for_last_30_days),
                avg.video_views = mean(video_views),
                avg.video_views_last_30_days = mean(video_views_for_the_last_30_days),
                avg.uploads = mean(uploads),
                avg.earnings = mean(avg_earnings),
                avg.Population = mean(Population),
                avg.Aggregated_GDP = mean(Aggregated_GDP),
                avg.channel_lifetime = mean(channel_lifetime))
  })
  
  # Table3 showing the clustered data
  output$clustdata <- renderTable({
    df_res <- cbind(cluster = cluster_results()[[1]], df_res)
    df_res <- df_res[order(df_res$cluster), ]
    df_res
  }, include.rownames = FALSE)
  
  # Decision Tree model building and plotting
  tree_model = eventReactive(
    eventExpr = input$createModel,
    if(input$fetur_set == "Feature Set 1"){
      valueExpr = makeTree(fV_dev, dev_data)
    }else{
      valueExpr = makeTree(fV_ig, ig_data)
    }
  )
  
  output$tree_plot = renderPlot(
    rpart.plot(tree_model(), roundint = FALSE)
  )
  
  
  output$table4 <- renderTable({
    auc_df
  }, include.rownames = FALSE)
  
  output$table5 <- renderTable({
    dev_df
  }, include.rownames = FALSE)
  
  output$table3 <- renderTable({
    if(input$fetur_set == "Feature Set 1"){
      pretty_perf_table(tree_model(), dev_data[devsel], dev_data[,outcome]==pos, 
                                      dev_test[devsel], dev_test[,outcome]==pos, 
                                      title = "DT Model - dev features")
    }else{
      pretty_perf_table(tree_model(), ig_data[igsel], ig_data[,outcome]==pos, 
                                      enc_test[igsel], enc_test[,outcome]==pos,
                                      title = "DT Model - IG features")
    }
  })
  
  
})

# Run the application 
shinyApp(ui = ui, server = server)
