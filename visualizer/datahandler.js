function RequestCurrentMap()
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleCurrentMap(xmlHttp.responseText);            
        }
    };
    
    xmlHttp.open("GET", "/v/curstate/" + current_step, true);
    xmlHttp.send();

    var xmlHttp2 = new XMLHttpRequest();
    xmlHttp2.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleCurrentExp(xmlHttp2.responseText);            
        }
    };
    
    xmlHttp2.open("GET", "/v/expinfo/", true);
    xmlHttp2.send();
    if(clicked != -1)
        RequestOutput();
}
function RequestOutput()
{
    latentoutput = undefined;
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleOutput(xmlHttp.responseText);            
        }
    };
    url = "/v/predicted/" + clicked + "/";
    for(var i = 0; i < 8; ++i)
        url += document.getElementById("value_l" + i).innerHTML + "/"
    xmlHttp.open("GET", url, true);
    xmlHttp.send();
}
function RequestCurrentAgent(target)
{/*
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleCurrentAgentResult(target, xmlHttp.responseText);            
        }
    };
    url = "/v/agentinfo/" + target + "/";
    xmlHttp.open("GET", url, true);
    xmlHttp.send();*/
}

function HandleCurrentMap(response)
{
    data = JSON.parse(response);
    vehicles = data["state"];
    /*
    impatiences = data["impatience"];

    if(clicked != -1)
    {
        document.getElementById("slider_impatience").value = impatiences[clicked]
    }
    */
    DrawCanvas();
}

function HandleCurrentExp(response)
{
    data = JSON.parse(response);
    max_step = data["max_step"];
    latent_length = data["latent_len"];

    document.getElementById("slider_step").max = max_step;
    for(var i = 0; i < 8; ++i)
    {
        if(i >= latent_length)
            document.getElementById("div_l" + i).hidden = true;
    }
    
    labels =[]
    for(var i = 0; i < max_step; ++i)
    {
        labels.push(i)
    }

    linechart.data.labels = labels;
    

}
function HandleOutput(response)
{
    data = JSON.parse(response);
    real_output = data["route"];
    latent_output = data["predicted"];
    latent_output_prob = data["action_prob"];
    DrawCanvas();
}
function HandleCurrentAgentResult(c, response)
{
    
    j = JSON.parse(response);
    datalist = [];
    const bordercolors = ["#1e90ff", "#ff1493", "#228b22", "#daa520", "#4B0082", "#4169E1", "#008B8B", "#006400"]
    for(var i = 0; i < 4; ++i)
    {
        datalist.push({
            label: (i < 4 ? "global" + i : "local" + (i-4)),
            data: [],
            borderColor: bordercolors[i],
            fill: false})
    }
    for(var i = 0; i < j["global_latent_history"].length; ++i)
    {
        datalist[0]["data"].push(j["global_latent_history"][i][0]);
        datalist[1]["data"].push(j["global_latent_history"][i][1]);
        datalist[2]["data"].push(j["global_latent_history"][i][2]);
        datalist[3]["data"].push(j["global_latent_history"][i][3]);
    }
    latents[c] = j["global_latent"]
    latent_data[c] = datalist;
    if(clicked == c)
    {
        linechart.data.datasets = datalist;
        linechart.update('none');

        MoveLatentToDefault();
    }
}
